"""
simulator.py  --  Part C: the general MiChroM simulator wrapper.

ONE simulator for every experiment in the project. What varies:
    - WHAT is simulated : a Model (Part A) -> any kernel, any coupling matrix,
                          any bead types. Also arbitrary extra forces.
    - WHICH phases run  : collapse / equilibration / production, in any subset.
    - WHERE it starts   : spring-spiral, or supplied initial coordinates.
    - WHAT is saved     : SaveSpec -> contact map, positions, velocities, forces.

What NEVER varies: the physics. The constants below reproduced the validated
k2 result (pooled saddle 2.087; re-validated end-to-end at 2.073). Do not change
them without re-running the acceptance test.

Design note (why one class, not two): the statics program and the dynamics
program use the SAME Langevin dynamics on the SAME landscape (this is the whole
point of Di Pierro 2018). Forking the simulator would create two copies of the
physics that can silently drift, after which statics and dynamics results would
no longer describe the same system. So: one physics implementation, swappable
inputs (Model) and swappable outputs (SaveSpec).

Storage/precision policy (matters for dynamics):
    - positions  : float32. NOT float16 -- you finite-difference these and
                   float16 destroys the short-lag signal.
    - velocities : SAVE THEM IF YOU MIGHT WANT THEM. They are stochastic and
                   NOT recoverable after the fact.
    - forces     : recoverable post hoc (deterministic given positions + the
                   exact System), so this is the one to drop under storage
                   pressure -- but only if run metadata is complete enough to
                   rebuild the System.
    Priority under pressure: sampling interval > velocities > forces.

All derived observables (contact maps, distances, enrichments, order parameters,
Koopman dictionaries) are computed POST HOC from saved state, never accumulated
during the run -- otherwise you discard the coordinates that later analyses
(e.g. gEDMD, which needs gradients/Hessians of observables) must differentiate
through.
"""

from __future__ import annotations
import contextlib
import io
import os
import gc
import glob
import json
import time
import shutil
import platform as _platform
from dataclasses import dataclass, asdict

import numpy as np


# =====================================================================
# Validated protocol constants -- DO NOT CHANGE without re-validating
# =====================================================================

N_STEPS_COLLAPSE = 200 * 1000
N_STEPS_EQUIL = 200 * 1000
N_STEPS_PRODUCTION = 3 * 10 ** 6
SAVE_INTERVAL = 1000
COLLAPSE_BLOCKSIZE = 3 * 10 ** 2

MU = 3.22
RC = 1.78
IC_DINIT = 3
IC_DEND = 500

KFB = 30.0
KA = 2.0
SELFAVOID_ECUT = 4.0
SELFAVOID_KREP = 20.0
SELFAVOID_R0 = 1.0

FLATBOTTOM_KR = 5 * 10 ** -3
FLATBOTTOM_NRAD = 15.0

TEMPERATURE = 1.0
TIMESTEP = 0.01


# =====================================================================
# Background stack (Part B): fixed physical terms, present during fitting
# =====================================================================

class BackgroundStack:
    """Fixed (specified-from-physics, not fit-to-map) terms. The compartment
    term is the only LEARNED part; everything here is background.

    `.key()` gives a signature so fit-environment can be checked against
    simulate-environment (the retraining-rule guard: Lambda is only valid in
    the environment it was trained in).
    """

    def __init__(self,
                 fene=True, angles=True, self_avoidance=True,
                 ideal_chromosome=True, phase1_confinement=True,
                 kFb=KFB, kA=KA,
                 ecut=SELFAVOID_ECUT, krep=SELFAVOID_KREP, r0=SELFAVOID_R0,
                 ic_mu=MU, ic_rc=RC, ic_dinit=IC_DINIT, ic_dend=IC_DEND,
                 conf_kR=FLATBOTTOM_KR, conf_nRad=FLATBOTTOM_NRAD,
                 extra_forces=None):
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
        # escape hatch for future environment terms (lamina, nucleolus,
        # crowding): list of callables f(sim) that add a force.
        self.extra_forces = list(extra_forces or [])

    def key(self):
        d = {k: v for k, v in self.__dict__.items() if k != "extra_forces"}
        d["n_extra_forces"] = len(self.extra_forces)
        return d

    def apply_pre_compartment(self, sim):
        """Backbone terms, added before the compartment term (validated order)."""
        if self.fene:
            sim.addFENEBonds(kFb=self.kFb)
        if self.angles:
            sim.addAngles(kA=self.kA)
        if self.self_avoidance:
            sim.addSelfAvoidance(Ecut=self.ecut, k_rep=self.krep, r0=self.r0)

    def apply_post_compartment(self, sim, confinement=None):
        conf = self.phase1_confinement if confinement is None else confinement
        if self.ideal_chromosome:
            sim.addIdealChromosome(mu=self.ic_mu, rc=self.ic_rc,
                                   dinit=self.ic_dinit, dend=self.ic_dend)
        if conf:
            sim.addFlatBottomHarmonic(kR=self.conf_kR, nRad=self.conf_nRad)
        for f in self.extra_forces:
            f(sim)


# =====================================================================
# Save policy
# =====================================================================

@dataclass
class SaveSpec:
    """What to persist from a run.

    contact_map : pooled contact probability map (the statics deliverable)
    trajectory  : per-frame positions, float32
    velocities  : per-frame velocities (NOT recoverable later -- save if unsure)
    forces      : per-frame forces (recoverable later from positions + System)
    interval    : steps between saved frames. Save uniformly at the finest
                  affordable interval, then subsample for long lags -- uniform
                  beats burst sampling because subsampling gives every lag free.
    monitor     : log radius of gyration each frame for run-health monitoring
                  only; never used as a scientific observable.
    """
    contact_map: bool = True
    trajectory: bool = False
    velocities: bool = False
    forces: bool = False
    interval: int = SAVE_INTERVAL
    monitor: bool = True

    @property
    def needs_manual_loop(self):
        """Reporter-based running only yields positions (cndb). Velocities,
        forces and per-frame monitoring require stepping manually and querying
        the Context."""
        return self.trajectory or self.velocities or self.forces or self.monitor


SAVE_MAPS_ONLY = SaveSpec(contact_map=True, trajectory=False, monitor=False)
SAVE_DYNAMICS = SaveSpec(contact_map=True, trajectory=True, velocities=True,
                         forces=False, interval=100)
SAVE_EVERYTHING = SaveSpec(contact_map=True, trajectory=True, velocities=True,
                           forces=True, interval=100)


# =====================================================================
# Contact map estimator -- identical to OpenMiChroM cndbTools.traj2HiC
# =====================================================================

def contact_probability(xyz, mu=MU, rc=RC):
    """Pooled contact probability map from frames (n_frames, n_beads, 3).

    f(r) = 0.5*(1 + tanh[mu*(rc - r)]), the SAME estimator as OpenMiChroM's
    cndbTools.traj2HiC (which averages over frames), so maps are directly
    comparable to every previously computed result.
    """
    from scipy.spatial import distance
    xyz = np.asarray(xyz)
    n = xyz.shape[1]
    P = np.zeros((n, n), dtype=np.float64)
    for f in range(xyz.shape[0]):
        d = distance.cdist(xyz[f], xyz[f], "euclidean")
        P += 0.5 * (1.0 + np.tanh(mu * (rc - d)))
    return P / xyz.shape[0]


def radius_of_gyration(x):
    c = x.mean(axis=0)
    return float(np.sqrt(((x - c) ** 2).sum(axis=1).mean()))


def observed_over_expected(P):
    """O/E: divide each diagonal by its mean."""
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
    """(<O/E>_AA <O/E>_BB) / <O/E>_AB^2."""
    def f(x, y):
        return np.nanmean(OE[np.ix_(x, y)])
    return (f(isA, isA) * f(isB, isB)) / f(isA, isB) ** 2


# =====================================================================
# Run metadata -- written on EVERY run
# =====================================================================

def collect_metadata(sim, background, save_spec, extra=None):
    """Everything needed to (a) reproduce the run and (b) rebuild the exact
    System so forces can be recomputed post hoc if they were not saved.
    This is the block that people get burned by omitting."""
    md = {}
    try:
        import openmm
        md["openmm_version"] = openmm.version.version
    except Exception:
        md["openmm_version"] = None
    try:
        import OpenMiChroM
        md["openmichrom_version"] = getattr(OpenMiChroM, "__version__", None)
    except Exception:
        md["openmichrom_version"] = None
    md["python"] = _platform.python_version()
    md["platform"] = _platform.platform()

    try:
        integ = sim.simulation.integrator
        md["integrator"] = type(integ).__name__
        for attr, key in (("getStepSize", "timestep"),
                          ("getFriction", "friction_gamma"),
                          ("getTemperature", "temperature")):
            if hasattr(integ, attr):
                try:
                    md[key] = str(getattr(integ, attr)())
                except Exception:
                    pass
    except Exception:
        pass

    try:
        system = sim.simulation.context.getSystem()
        md["n_particles"] = system.getNumParticles()
        md["particle_mass_0"] = str(system.getParticleMass(0))
        forces = [type(system.getForce(i)).__name__
                  for i in range(system.getNumForces())]
        md["forces"] = forces
        md["cm_motion_remover"] = any("CMMotionRemover" in f for f in forces)
    except Exception:
        pass

    try:
        md["compute_platform"] = \
            sim.simulation.context.getPlatform().getName()
    except Exception:
        pass

    md["background"] = background.key()
    md["save_spec"] = asdict(save_spec)
    md["units_note"] = ("MiChroM reduced units. Positions/velocities/forces as "
                        "returned by OpenMM (labelled nm, nm/ps, kJ/mol/nm); "
                        "treat as reduced units, not SI.")
    if extra:
        md.update(extra)
    return md


# =====================================================================
# The simulator
# =====================================================================

class Simulator:
    """General MiChroM runner.

    seq_file  : ChromSeq file. Registers bead types -- OpenMiChroM requires
                createSpringSpiral for type registration even when the initial
                coordinates are subsequently overridden.
    out_dir   : outputs (resumable): maps/, traj/, meta/.
    background: BackgroundStack.
    platform  : 'cuda' | 'opencl' | 'cpu'.
    """

    def __init__(self, seq_file, out_dir, background=None, platform="cuda",
                 work_root=None, conv_threshold=0.9,
                 temperature=TEMPERATURE, timestep=TIMESTEP, quiet=False):
        self.seq_file = seq_file
        self.out_dir = out_dir
        self.background = background or BackgroundStack()
        self.platform = platform
        self.work_root = work_root or os.path.join(out_dir, "_scratch")
        self.conv_threshold = conv_threshold
        self.temperature = temperature
        self.timestep = timestep
        # OpenMiChroM reprints its full banner and per-force energy table on
        # EVERY MiChroM() construction. Across an optimizer fit that is one
        # block per replica per iteration -- thousands of screens of text that
        # bury the fit's own log. `quiet` suppresses it by capturing stdout
        # during construction only; OpenMiChroM itself is untouched, and the
        # captured text is kept on `.last_banner` if it is ever needed.
        self.quiet = quiet
        self.last_banner = ""
        for sub in ("maps", "traj", "meta"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        os.makedirs(self.work_root, exist_ok=True)

    @contextlib.contextmanager
    def _muffle(self):
        """Capture library chatter during system construction if quiet."""
        if not self.quiet:
            yield
            return
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                yield
        finally:
            self.last_banner = buf.getvalue()

    # ---------- internals ----------
    def sequence_names(self):
        """Per-bead names from the ChromSeq file (2nd column). These are the
        names OpenMiChroM's addCustomTypes expects in the .ff header."""
        if getattr(self, "_seq_names", None) is None:
            self._seq_names = [l.split()[1] for l in open(self.seq_file)
                               if l.strip()]
        return self._seq_names

    def _scratch(self, name):
        return os.path.join(self.work_root, name)

    def _cleanup_scratch(self, name):
        try:
            import h5py
            for o in gc.get_objects():
                try:
                    if isinstance(o, h5py.File) and o.id.valid:
                        o.close()
                except Exception:
                    pass
        except Exception:
            pass
        shutil.rmtree(self._scratch(name), ignore_errors=True)

    def _build(self, name, model=None, types_table=None, initial_coords=None,
               confinement=None, seed=None, perturb=True):
        """Create a MiChroM object: background + compartment term + IC/conf.

        model         : Part-A Model; its coupling matrix is written to a .ff
                        and loaded via addCustomTypes.
        types_table   : path to an existing .ff (alternative to `model`).
        initial_coords: (N,3) starting configuration; None -> spring spiral.
        """
        from OpenMiChroM.ChromDynamics import MiChroM

        os.makedirs(self._scratch(name), exist_ok=True)
        with self._muffle():
            sim = MiChroM(temperature=self.temperature, timeStep=self.timestep)
            sim.setup(platform=self.platform)
            sim.saveFolder(self._scratch(name))

        with self._muffle():
            # Type registration REQUIRES createSpringSpiral even when the
            # coordinates are then overridden (OpenMiChroM design constraint).
            struct = sim.createSpringSpiral(ChromSeq=self.seq_file, isRing=False)
            if initial_coords is None:
                s = np.asarray(struct, float)
                if perturb and seed is not None:
                    s = s + np.random.default_rng(seed).normal(0, 0.05, s.shape)
                sim.loadStructure(s, center=True)
            else:
                # center=False preserves the exact incoming configuration
                sim.loadStructure(np.asarray(initial_coords, float), center=False)

            self.background.apply_pre_compartment(sim)

            ff_path = None
            if model is not None:
                ff_path = os.path.join(self._scratch(name), "_coupling.ff")
                # The .ff header MUST match the names in the ChromSeq file.
                # A per-BEAD model (one row per bead) takes the bead names from
                # the sequence file; a per-TYPE model keeps its own type names.
                seq_names = self.sequence_names()
                bead_names = seq_names if model.N == len(seq_names) else None
                model.write_ff(ff_path, bead_names=bead_names)
                sim.addCustomTypes(TypesTable=ff_path, mu=MU, rc=RC)
            elif types_table is not None:
                sim.addCustomTypes(TypesTable=types_table, mu=MU, rc=RC)
            else:
                sim.addTypetoType(mu=MU, rc=RC)

            self.background.apply_post_compartment(sim, confinement=confinement)
            sim.createSimulation()
        return sim, ff_path

    # ---------- the run ----------
    def run_replica(self, cond, seed, model=None, types_table=None,
                    initial_coords=None,
                    phases=("collapse", "equil", "production"),
                    n_production=N_STEPS_PRODUCTION,
                    save_spec=None, verbose=True, overwrite=False):
        """Run one replica.

        phases : subset/order of ('collapse','equil','production').
                 Statics default = all three (the validated protocol).
                 QUENCH runs = ('production',) with initial_coords supplied.
                 Equilibration MUST NOT be run for a quench, or the transient
                 being measured is destroyed.
        """
        save_spec = save_spec or SAVE_MAPS_ONLY
        name = f"{cond}_rep{seed}"
        mapfile = os.path.join(self.out_dir, "maps", f"{name}.npy")
        trajfile = os.path.join(self.out_dir, "traj", f"{name}.npz")
        metafile = os.path.join(self.out_dir, "meta", f"{name}.json")

        already = ((os.path.exists(mapfile) or not save_spec.contact_map)
                   and (os.path.exists(trajfile) or not save_spec.trajectory))
        if already and not overwrite:
            if verbose:
                print("skip", name, flush=True)
            return None

        self._cleanup_scratch(name)
        t0 = time.time()

        use_confinement = ("collapse" in phases)
        sim, _ = self._build(name, model=model, types_table=types_table,
                             initial_coords=initial_coords, seed=seed,
                             confinement=use_confinement)

        meta = collect_metadata(sim, self.background, save_spec, extra={
            "cond": cond, "seed": seed, "phases": list(phases),
            "n_production": int(n_production),
            "n_steps_collapse": N_STEPS_COLLAPSE if "collapse" in phases else 0,
            "n_steps_equil": N_STEPS_EQUIL if "equil" in phases else 0,
            "seq_file": self.seq_file,
            "rng_seed": seed,
            "initial_coords_supplied": initial_coords is not None,
            "is_quench": (initial_coords is not None
                          and tuple(phases) == ("production",)),
        })

        with self._muffle():
            if "collapse" in phases:
                sim.run(nsteps=N_STEPS_COLLAPSE, checkSystem=True, report=True,
                        blockSize=COLLAPSE_BLOCKSIZE)
                if self.background.phase1_confinement:
                    sim.removeFlatBottomHarmonic()
            if "equil" in phases:
                sim.run(nsteps=N_STEPS_EQUIL, report=True)

        frames = vels = forces = None
        rg_trace = []
        if "production" in phases:
            if save_spec.needs_manual_loop:
                frames, vels, forces, rg_trace = self._production_manual(
                    sim, n_production, save_spec, verbose=verbose)
            else:
                frames = self._production_reporters(
                    sim, n_production, name, save_spec)

        out = {}
        if frames is not None and save_spec.contact_map:
            P = contact_probability(frames)
            np.save(mapfile, P.astype(np.float32))
            conv = self._convergence(frames)
            np.save(os.path.join(self.out_dir, "maps", f"{name}.conv.npy"),
                    np.array([conv]))
            meta["convergence"] = conv
            out["map"] = mapfile
        if save_spec.trajectory and frames is not None:
            arrays = {"xyz": frames.astype(np.float32)}
            if save_spec.velocities and vels is not None:
                arrays["vel"] = vels.astype(np.float32)
            if save_spec.forces and forces is not None:
                arrays["frc"] = forces.astype(np.float32)
            arrays["frame_steps"] = (np.arange(1, frames.shape[0] + 1,
                                               dtype=np.int64)
                                     * save_spec.interval)
            np.savez_compressed(trajfile, **arrays)
            out["traj"] = trajfile
        if rg_trace:
            meta["rg_trace"] = rg_trace
        meta["wall_minutes"] = (time.time() - t0) / 60.0
        with open(metafile, "w") as fh:
            json.dump(meta, fh, indent=1, default=str)
        out["meta"] = metafile

        self._close_reporters(sim)
        self._cleanup_scratch(name)

        if verbose:
            c = meta.get("convergence", float("nan"))
            flag = "" if (not np.isfinite(c) or c >= self.conv_threshold) \
                else f"  *** LOW CONVERGENCE {c:.3f} ***"
            print(f"{name}: {meta['wall_minutes']:.1f} min  conv={c:.3f}{flag}",
                  flush=True)
        return out

    def _production_manual(self, sim, n_steps, save_spec, verbose=True):
        """Step in blocks and query the Context. Required for velocities and
        forces. Physically identical to an uninterrupted run."""
        n_blocks = int(n_steps // save_spec.interval)
        pos, vel, frc, rg = [], [], [], []
        need_v, need_f = save_spec.velocities, save_spec.forces
        for b in range(n_blocks):
            sim.run(nsteps=save_spec.interval, report=False)
            state = sim.simulation.context.getState(
                getPositions=True, getVelocities=need_v, getForces=need_f)
            x = np.asarray(state.getPositions(asNumpy=True)._value,
                           dtype=np.float32)
            pos.append(x)
            if need_v:
                vel.append(np.asarray(state.getVelocities(asNumpy=True)._value,
                                      dtype=np.float32))
            if need_f:
                frc.append(np.asarray(state.getForces(asNumpy=True)._value,
                                      dtype=np.float32))
            if save_spec.monitor:
                rg.append(radius_of_gyration(x))
            if verbose and n_blocks >= 10 and b % max(1, n_blocks // 10) == 0:
                print(f"    block {b}/{n_blocks}", flush=True)
        return (np.array(pos),
                np.array(vel) if need_v else None,
                np.array(frc) if need_f else None,
                rg)

    def _production_reporters(self, sim, n_steps, name, save_spec):
        """Validated statics path: cndb reporters, positions only."""
        sim.createReporters(statistics=True, traj=True, outputName=name,
                            trajFormat="cndb", energyComponents=True,
                            interval=save_spec.interval)
        sim.run(nsteps=n_steps, report=True)
        self._close_reporters(sim)
        from OpenMiChroM.CndbTools import cndbTools
        cndb = cndbTools()
        path = glob.glob(os.path.join(self._scratch(name), "*.cndb"))[0]
        cndb.load(path)
        nums = sorted(int(k) for k in cndb.cndb.keys() if k.isdigit())
        xyz = cndb.xyz(frames=nums)
        try:
            cndb.cndb.close()
        except Exception:
            pass
        return np.asarray(xyz, dtype=np.float32)

    @staticmethod
    def _close_reporters(sim):
        try:
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
        except Exception:
            pass

    def _convergence(self, frames):
        nf = len(frames)
        if nf < 4:
            return float("nan")
        h1 = contact_probability(frames[: nf // 2])
        h2 = contact_probability(frames[nf // 2:])
        iu = np.triu_indices(frames.shape[1], 3)
        a, b = h1[iu], h2[iu]
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 \
            else float("nan")

    # ---------- ensembles ----------
    def run_ensemble(self, cond, model=None, types_table=None, n_replicas=20,
                     save_spec=None, verbose=True, **kw):
        for s in range(n_replicas):
            self.run_replica(cond, s, model=model, types_table=types_table,
                             save_spec=save_spec, verbose=verbose, **kw)
        return self.pool(cond, n_replicas)

    def pool(self, cond, n_replicas=None):
        maps_dir = os.path.join(self.out_dir, "maps")
        files = [f for f in sorted(
            glob.glob(os.path.join(maps_dir, f"{cond}_rep*.npy")),
            key=lambda p: int(p.split("rep")[-1].split(".")[0]))
            if ".conv." not in f]
        if n_replicas is not None:
            files = files[:n_replicas]
        assert files, f"no maps found for {cond}"
        acc, low = None, []
        for f in files:
            P = np.load(f).astype(np.float64)
            acc = P if acc is None else acc + P
            cf = f.replace(".npy", ".conv.npy")
            if os.path.exists(cf):
                c = float(np.load(cf)[0])
                if np.isfinite(c) and c < self.conv_threshold:
                    low.append((os.path.basename(f), round(c, 3)))
        if low:
            print(f"WARNING: {len(low)} low-convergence replicas: {low}")
        return acc / len(files)

    # ---------- trajectory access ----------
    def load_trajectory(self, cond, seed):
        p = os.path.join(self.out_dir, "traj", f"{cond}_rep{seed}.npz")
        with np.load(p) as z:
            return {k: z[k] for k in z.files}

    def load_metadata(self, cond, seed):
        with open(os.path.join(self.out_dir, "meta",
                               f"{cond}_rep{seed}.json")) as fh:
            return json.load(fh)

    def final_frames(self, cond, n_replicas):
        """Last frame of each replica -- the clean way to seed a quench: one
        INDEPENDENT start per replica. (Drawing many frames from one replica
        gives correlated starts and inflates apparent ensemble size.)"""
        out = []
        for s in range(n_replicas):
            try:
                out.append(self.load_trajectory(cond, s)["xyz"][-1])
            except Exception:
                pass
        return np.array(out)

    # ---------- quench: restart-based (always works) ----------
    def quench_replica(self, cond, seed, model_B, initial_coords,
                       n_production=N_STEPS_PRODUCTION, save_spec=None,
                       verbose=True):
        """Instantaneous H_A -> H_B quench by restarting under H_B from an
        H_A configuration.

        Position-continuous, velocity-DISCONTINUOUS: createSimulation draws
        fresh Maxwell-Boltzmann velocities, so the first frames after the swap
        contain a thermal transient that is not structural relaxation -- flag
        or discard them. Use the in-place path (below) for velocity continuity
        if verify_inplace_quench() reports support.
        """
        save_spec = save_spec or SAVE_DYNAMICS
        return self.run_replica(cond, seed, model=model_B,
                                initial_coords=initial_coords,
                                phases=("production",),
                                n_production=n_production,
                                save_spec=save_spec, verbose=verbose)


# =====================================================================
# In-place quench (velocity-continuous) -- SUPPORT MUST BE VERIFIED
# =====================================================================

def _custom_nonbonded_forces(system):
    import openmm
    out = []
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, openmm.CustomNonbondedForce):
            try:
                n_tab = f.getNumTabulatedFunctions()
            except Exception:
                n_tab = 0
            out.append((i, f, n_tab))
    return out


def swap_coupling_in_context(sim, new_matrix):
    """Attempt to overwrite the type-type coupling in a LIVE Context.

    Returns True if a matching Discrete2DFunction was found, updated, and
    pushed via updateParametersInContext. Velocity-continuous: this is a true
    instantaneous H_A -> H_B switch with no re-initialisation.

    NOT guaranteed to work -- depends on how OpenMiChroM builds the force.
    Run verify_inplace_quench() once to find out. Falls back to
    Simulator.quench_replica() (restart-based) if unsupported.
    """
    import openmm
    ctx = sim.simulation.context
    system = ctx.getSystem()
    M = np.asarray(new_matrix, float)
    done = False
    for idx, f, n_tab in _custom_nonbonded_forces(system):
        for t in range(n_tab):
            fn = f.getTabulatedFunction(t)
            if isinstance(fn, openmm.Discrete2DFunction):
                xs, ys, _ = fn.getFunctionParameters()
                if xs * ys == M.size:
                    # OpenMM Discrete2DFunction is column-major in x
                    fn.setFunctionParameters(xs, ys, M.T.flatten().tolist())
                    f.updateParametersInContext(ctx)
                    done = True
    return done


def verify_inplace_quench(seq_file, model_A, model_B, platform="cuda",
                          out_dir=None, background=None):
    """Empirically determine whether the coupling matrix can be swapped in a
    live Context. RUN THIS ONCE before relying on the in-place path.

    Procedure: build under model_A, record potential energy, attempt the swap
    to model_B, re-read the energy, and report whether it changed (i.e. whether
    the swap actually took effect).

    Returns a report dict. If report['supported'] is False, use the
    restart-based quench_replica -- it always works.
    """
    out_dir = out_dir or "./_verify_quench"
    wrap = Simulator(seq_file, out_dir, background=background,
                     platform=platform)
    sim, _ = wrap._build("verify", model=model_A, perturb=False)
    ctx = sim.simulation.context
    system = ctx.getSystem()

    report = {"custom_nonbonded_forces": [], "supported": False, "notes": []}
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy()._value
    report["energy_before"] = e0

    for idx, f, n_tab in _custom_nonbonded_forces(system):
        report["custom_nonbonded_forces"].append({
            "index": idx,
            "energy_expression": f.getEnergyFunction()[:200],
            "n_tabulated": n_tab,
            "n_per_particle_params": f.getNumPerParticleParameters(),
        })

    try:
        ok = swap_coupling_in_context(sim, model_B.coupling_matrix())
    except Exception as exc:
        ok = False
        report["notes"].append(f"swap raised: {exc}")

    if ok:
        e1 = ctx.getState(getEnergy=True).getPotentialEnergy()._value
        report["energy_after"] = e1
        report["energy_changed"] = abs(e1 - e0) > 1e-9
        report["supported"] = report["energy_changed"]
        if not report["energy_changed"]:
            report["notes"].append(
                "update succeeded but energy did not change -- the coupling "
                "may not live in the tabulated function that was overwritten.")
    else:
        report["notes"].append(
            "no swappable Discrete2DFunction matching the coupling matrix "
            "shape was found; use restart-based quench_replica().")
    wrap._cleanup_scratch("verify")
    return report


# =====================================================================
# Pilot: the check that determines whether generator-based methods are viable
# =====================================================================

PILOT = r'''
OVERDAMPED / DATA-INTEGRITY PILOT -- run BEFORE any production dynamics ensemble.

Saves positions, velocities and forces EVERY STEP on a SMALL system and checks:

  1. OVERDAMPED CONDITION -- does F/(gamma*m) track the actual velocities?
     If yes, dynamics are effectively overdamped and generator-based methods
     (gEDMD) are well founded. If no, the inertial term matters and the
     underdamped formulation is needed. THIS IS THE QUESTION THAT COULD
     INVALIDATE THE METHOD, and it costs one short run to answer.
  2. FORCE SANITY -- finite, no blow-ups, sensible magnitudes.
  3. ROUND-TRIP -- saved positions reload without precision loss that would
     corrupt finite differences.

    from chromatin_potential.simulator import Simulator, SaveSpec
    import numpy as np

    spec = SaveSpec(contact_map=False, trajectory=True, velocities=True,
                    forces=True, interval=1)               # every step
    sim = Simulator(seq_file=SMALL_SEQ, out_dir=OUT, platform='cuda')
    sim.run_replica('pilot', 0, model=m,
                    phases=('collapse','equil','production'),
                    n_production=2000, save_spec=spec)

    d, md = sim.load_trajectory('pilot', 0), sim.load_metadata('pilot', 0)
    x, v, f = d['xyz'], d['vel'], d['frc']

    gamma = float(str(md['friction_gamma']).split()[0])
    mass  = float(str(md['particle_mass_0']).split()[0])
    pred  = f / (gamma * mass)
    print('corr(v, F/(gamma m)) =', np.corrcoef(pred.ravel(), v.ravel())[0,1])
    print('magnitude ratio      =', np.linalg.norm(pred)/np.linalg.norm(v))
    print('|F| mean/max         =', np.abs(f).mean(), np.abs(f).max(),
          'finite:', np.isfinite(f).all())
    print('positions dtype      =', x.dtype, 'finite:', np.isfinite(x).all())

Interpretation: correlation near 1 and ratio near 1 => strongly overdamped,
gEDMD on positions alone is justified. Low correlation => inertia matters; use
the underdamped generator or include velocities in the state vector.
'''

if __name__ == "__main__":
    print(PILOT)
