"""
simulator.py  --  Part C: the simulator wrapper.

Wraps OpenMiChroM's MiChroM object into a reusable runner that takes:
    - a Model (Part A)         -> the compartment coupling matrix (via .ff)
    - a BackgroundStack (Part B) -> fixed physical terms present during sim/fit
    - a chain topology         -> which beads belong to which chain (one for now,
                                  architected for many so multi-chromosome is an
                                  add, not a rewrite)

and runs the VALIDATED three-phase protocol, pools replicas, returns the pooled
contact map, with a CONVERGENCE GATE so no run silently returns undersampled
garbage.

Every physics parameter here is copied VERBATIM from the runner that produced the
completed k2 result (saddle 2.087). Do not change kFb, kA, self-avoidance, mu, rc,
the IC range, the flat-bottom radius, or the step counts without re-validating.

This module DOES need OpenMiChroM + a GPU. Unlike model.py it is not pure-NumPy.
Acceptance test (see bottom / tests): feed the k2 model's coupling matrix, run 20
replicas, pooled saddle must be 2.087 +/- replica noise.
"""

from __future__ import annotations
import os
import gc
import glob
import shutil
import numpy as np


# =====================================================================
# Validated protocol constants -- DO NOT CHANGE without re-validating
# =====================================================================

N_STEPS_COLLAPSE = 200 * 1000      # phase 1: confined collapse (flat-bottom on)
N_STEPS_EQUIL = 200 * 1000         # phase 2: unconfined equilibration (no reporters)
N_STEPS_PRODUCTION = 3 * 10 ** 6   # phase 3: production (reporters on)
SAVE_INTERVAL = 1000               # -> 3000 frames/replica
COLLAPSE_BLOCKSIZE = 3 * 10 ** 2

# compartment / IC kernel
MU = 3.22
RC = 1.78
IC_DINIT = 3
IC_DEND = 500

# backbone / excluded volume
KFB = 30.0
KA = 2.0
SELFAVOID_ECUT = 4.0
SELFAVOID_KREP = 20.0
SELFAVOID_R0 = 1.0

# confinement (phase-1 flat-bottom)
FLATBOTTOM_KR = 5 * 10 ** -3
FLATBOTTOM_NRAD = 15.0


# =====================================================================
# Background stack (Part B, minimal here; a full registry lives in background.py)
# =====================================================================

class BackgroundStack:
    """The fixed physical terms present during simulation AND fitting.

    Minimal version: backbone + excluded volume + ideal chromosome + phase-1
    confinement, matching the validated runner. Environment modules
    (confinement-as-crowding, other chains, lamina, nucleolus) are added here
    later; the point is that fit and simulate use the SAME stack so effects don't
    leak into the learned Lambda.

    For now it is a config object the simulator reads; each flag maps to one
    OpenMiChroM force call, applied in the validated order.
    """

    def __init__(self,
                 fene=True, angles=True, self_avoidance=True,
                 ideal_chromosome=True, phase1_confinement=True,
                 kFb=KFB, kA=KA,
                 ecut=SELFAVOID_ECUT, krep=SELFAVOID_KREP, r0=SELFAVOID_R0,
                 ic_mu=MU, ic_rc=RC, ic_dinit=IC_DINIT, ic_dend=IC_DEND,
                 conf_kR=FLATBOTTOM_KR, conf_nRad=FLATBOTTOM_NRAD):
        self.fene = fene
        self.angles = angles
        self.self_avoidance = self_avoidance
        self.ideal_chromosome = ideal_chromosome
        self.phase1_confinement = phase1_confinement
        self.kFb = kFb
        self.kA = kA
        self.ecut = ecut
        self.krep = krep
        self.r0 = r0
        self.ic_mu = ic_mu
        self.ic_rc = ic_rc
        self.ic_dinit = ic_dinit
        self.ic_dend = ic_dend
        self.conf_kR = conf_kR
        self.conf_nRad = conf_nRad

    def key(self):
        """A short hashable signature so the simulator can enforce that fit-env
        and sim-env match (the retraining-rule guard)."""
        return (self.fene, self.angles, self.self_avoidance,
                self.ideal_chromosome, self.phase1_confinement,
                self.kFb, self.kA, self.ecut, self.krep, self.r0,
                self.ic_mu, self.ic_rc, self.ic_dinit, self.ic_dend,
                self.conf_kR, self.conf_nRad)


# =====================================================================
# Convergence gate
# =====================================================================

def convergence_metric(frames_xyz, n_beads):
    """First-half vs second-half contact-map correlation. ~1.0 = converged.

    frames_xyz : (n_frames, n_beads, 3) trajectory
    Returns Pearson r between the contact maps of the two halves (upper triangle,
    separation > 2). Low r => the ensemble has not equilibrated / is undersampled.
    """
    from OpenMiChroM.CndbTools import cndbTools
    cndb = cndbTools()
    nf = len(frames_xyz)
    if nf < 4:
        return np.nan
    h1 = cndb.traj2HiC(frames_xyz[: nf // 2])
    h2 = cndb.traj2HiC(frames_xyz[nf // 2:])
    iu = np.triu_indices(n_beads, 3)
    a, b = h1[iu].astype(float), h2[iu].astype(float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


# =====================================================================
# The simulator wrapper
# =====================================================================

class Simulator:
    """Runs the validated three-phase protocol for a given Model + background.

    Parameters
    ----------
    seq_file    : path to the per-bead sequence file (ChromSeq) OpenMiChroM reads
                  to build the initial spring-spiral. For a per-bead custom-types
                  model this is the file whose 2nd column lists the bead 'type'
                  names matching the .ff header.
    out_dir     : where pooled per-replica maps are written (resumable).
    background  : BackgroundStack (defaults to the validated minimal stack).
    platform    : 'cuda' (default), 'opencl', or 'cpu'.
    work_root   : scratch dir for cndb/trajectory files (deleted per replica).
    conv_threshold : minimum first/second-half map correlation to accept a run;
                     runs below this are flagged (not silently kept).
    """

    def __init__(self, seq_file, out_dir, background=None, platform="cuda",
                 work_root=None, conv_threshold=0.9):
        self.seq_file = seq_file
        self.out_dir = out_dir
        self.background = background or BackgroundStack()
        self.platform = platform
        self.work_root = work_root or os.path.join(out_dir, "_scratch")
        self.conv_threshold = conv_threshold
        os.makedirs(os.path.join(out_dir, "maps"), exist_ok=True)
        os.makedirs(self.work_root, exist_ok=True)

    # ---- internals ----
    def _scratch(self, name):
        return os.path.join(self.work_root, name)

    def _cleanup(self, name):
        import h5py
        for o in gc.get_objects():
            try:
                if isinstance(o, h5py.File) and o.id.valid:
                    o.close()
            except Exception:
                pass
        shutil.rmtree(self._scratch(name), ignore_errors=True)

    def _frames_of(self, cndb_path):
        from OpenMiChroM.CndbTools import cndbTools
        cndb = cndbTools()
        cndb.load(cndb_path)
        nums = sorted(int(k) for k in cndb.cndb.keys() if k.isdigit())
        xyz = cndb.xyz(frames=nums)
        try:
            cndb.cndb.close()
        except Exception:
            pass
        return xyz

    # ---- the run ----
    def run_replica(self, cond, seed, types_table, verbose=True):
        """Run one replica. Resumable: skips if the map already exists.

        cond         : condition label (e.g. 'k2')
        seed         : replica index (also RNG seed for the initial perturbation)
        types_table  : path to the .ff file (Model.write_ff output), or None to
                       use MiChroM's built-in addTypetoType (the N0 arm).
        Returns the per-replica contact map (or None if skipped).
        """
        from OpenMiChroM.ChromDynamics import MiChroM

        name = f"{cond}_rep{seed}"
        mapfile = os.path.join(self.out_dir, "maps", f"{name}.npy")
        if os.path.exists(mapfile):
            if verbose:
                print("skip", name, flush=True)
            return None
        self._cleanup(name)
        bg = self.background

        sim = MiChroM(temperature=1.0, timeStep=0.01)
        sim.setup(platform=self.platform)
        sim.saveFolder(self._scratch(name))

        struct = sim.createSpringSpiral(ChromSeq=self.seq_file, isRing=False)
        struct = struct + np.random.default_rng(seed).normal(0, 0.05, struct.shape)
        sim.loadStructure(struct, center=True)

        # --- background stack, applied in the validated order ---
        if bg.fene:
            sim.addFENEBonds(kFb=bg.kFb)
        if bg.angles:
            sim.addAngles(kA=bg.kA)
        if bg.self_avoidance:
            sim.addSelfAvoidance(Ecut=bg.ecut, k_rep=bg.krep, r0=bg.r0)

        # --- compartment term (the learned part) ---
        if types_table is None:
            sim.addTypetoType(mu=MU, rc=RC)
        else:
            sim.addCustomTypes(TypesTable=types_table, mu=MU, rc=RC)

        if bg.ideal_chromosome:
            sim.addIdealChromosome(mu=bg.ic_mu, rc=bg.ic_rc,
                                   dinit=bg.ic_dinit, dend=bg.ic_dend)
        if bg.phase1_confinement:
            sim.addFlatBottomHarmonic(kR=bg.conf_kR, nRad=bg.conf_nRad)

        sim.createSimulation()

        # phase 1: confined collapse
        sim.run(nsteps=N_STEPS_COLLAPSE, checkSystem=True, report=True,
                blockSize=COLLAPSE_BLOCKSIZE)
        # phase 2: unconfined equilibration (no reporters)
        if bg.phase1_confinement:
            sim.removeFlatBottomHarmonic()
        sim.run(nsteps=N_STEPS_EQUIL, report=True)
        # phase 3: production
        sim.createReporters(statistics=True, traj=True, outputName=name,
                            trajFormat="cndb", energyComponents=True,
                            interval=SAVE_INTERVAL)
        sim.run(nsteps=N_STEPS_PRODUCTION, report=True)

        # close reporter file handles (Windows will not delete open files)
        for r in sim.simulation.reporters:
            if hasattr(r, "_out"):
                try:
                    r._out.close()
                except Exception:
                    pass
            if hasattr(r, "storage") and isinstance(r.storage, list):
                for it in r.storage:
                    try:
                        it.close()
                    except Exception:
                        pass

        from OpenMiChroM.CndbTools import cndbTools
        cndb = cndbTools()
        cndb_file = glob.glob(os.path.join(self._scratch(name), "*.cndb"))[0]
        frames = self._frames_of(cndb_file)
        n_beads = frames.shape[1]

        # convergence gate
        conv = convergence_metric(frames, n_beads)
        P = cndb.traj2HiC(frames)
        np.save(mapfile, P.astype(np.float16))
        # record convergence alongside the map
        np.save(os.path.join(self.out_dir, "maps", f"{name}.conv.npy"),
                np.array([conv]))
        shutil.rmtree(self._scratch(name), ignore_errors=True)

        flag = "" if (np.isnan(conv) or conv >= self.conv_threshold) \
            else f"  *** LOW CONVERGENCE {conv:.3f} < {self.conv_threshold} ***"
        if verbose:
            print(f"saved {mapfile}  conv={conv:.3f}{flag}", flush=True)
        return P

    def run_ensemble(self, cond, types_table, n_replicas=20, verbose=True):
        """Run n_replicas (resumable) and return the pooled contact map plus the
        per-replica convergence values."""
        import time
        t0 = time.time()
        for s in range(n_replicas):
            self.run_replica(cond, s, types_table, verbose=verbose)
            if verbose:
                print(f"--- {cond} seed {s} done, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        return self.pool(cond, n_replicas)

    def pool(self, cond, n_replicas=None):
        """Pool existing per-replica maps into one contact map. Reports any
        replicas flagged for low convergence."""
        maps_dir = os.path.join(self.out_dir, "maps")
        files = sorted(
            glob.glob(os.path.join(maps_dir, f"{cond}_rep*.npy")),
            key=lambda p: int(p.split("rep")[-1].split(".")[0]))
        files = [f for f in files if ".conv." not in f]
        if n_replicas is not None:
            files = files[:n_replicas]
        assert files, f"no maps found for {cond}"
        acc, n = None, 0
        low = []
        for f in files:
            P = np.load(f).astype(np.float64)
            acc = P if acc is None else acc + P
            n += 1
            convf = f.replace(".npy", ".conv.npy")
            if os.path.exists(convf):
                c = float(np.load(convf)[0])
                if not np.isnan(c) and c < self.conv_threshold:
                    low.append((os.path.basename(f), c))
        pooled = acc / n
        if low:
            print(f"WARNING: {len(low)} low-convergence replicas in pool: {low}")
        return pooled


# =====================================================================
# Analysis helpers (metric definitions used by the acceptance test)
# =====================================================================

def observed_over_expected(P):
    """O/E: divide each diagonal by its mean. P is an (N,N) contact map."""
    N = P.shape[0]
    sep = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    v = np.asarray(P, float).ravel()
    sepf = sep.ravel()
    g = np.isfinite(v)
    s = np.bincount(sepf[g], weights=v[g], minlength=N)
    cc = np.bincount(sepf[g], minlength=N)
    e = np.divide(s, cc, out=np.full(N, np.nan), where=cc > 0)[sepf]
    O = np.divide(v, e, out=np.full_like(v, np.nan),
                  where=np.isfinite(e) & (e > 0) & g)
    return O.reshape(N, N)


def saddle_strength(OE, isA, isB):
    """(<O/E>_AA <O/E>_BB) / <O/E>_AB^2, bins masked by true A/B labels."""
    def f(x, y):
        return np.nanmean(OE[np.ix_(x, y)])
    return (f(isA, isA) * f(isB, isB)) / f(isA, isB) ** 2


# =====================================================================
# Acceptance test (documented; run in a GPU notebook, not here)
# =====================================================================

ACCEPTANCE = """
Part C acceptance test (run in a GPU notebook):

    from chromatin_potential.model import Model, _A_FULL, _L_FULL
    from chromatin_potential.simulator import (
        Simulator, observed_over_expected, saddle_strength)
    import numpy as np

    # 1. build the k2 per-bead potential from the Model (Part A)
    m_types = Model.from_type_matrix(_A_FULL, _L_FULL, k=2)
    labels  = [l.split()[1] for l in open(SEQ) if l.strip()]   # chr10 bead labels
    m_bead  = m_types.expand_to_beads(labels)
    m_bead.write_ff(f'{OUT}/k2_from_model.ff', bead_names=[f't{i:05d}' for i in range(len(labels))])

    # 2. simulate 20 replicas
    sim = Simulator(seq_file=PERBEAD_SEQ, out_dir=OUT, platform='cuda')
    pooled = sim.run_ensemble('k2model', f'{OUT}/k2_from_model.ff', n_replicas=20)

    # 3. saddle must reproduce the completed k2 result: 2.087 +/- replica noise
    labels = np.array(labels)
    isA = np.isin(labels, ['A1','A2']); isB = np.isin(labels, ['B1','B2','B3','B4'])
    OE  = observed_over_expected(pooled)
    print('k2-via-machinery saddle =', saddle_strength(OE, isA, isB), '(expect ~2.087)')

This closes the loop: Part A emits the matrix, Part C simulates it, and together
they reproduce the completed k2 result through the new machinery.
"""

if __name__ == "__main__":
    print(ACCEPTANCE)