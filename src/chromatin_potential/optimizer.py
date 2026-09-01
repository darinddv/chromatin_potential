"""
optimizer.py  --  fitting theta = (C, Lambda) to a target contact map.

The maximum-entropy gradient is nearly free: for a coupling that multiplies the
contact function f(r_ij), the derivative of the objective w.r.t. that coupling is
just the difference between simulated and target contact probability

    G_ij = phi_exp_ij - phi_sim_ij

(NOTE THE SIGN: the coupling multiplies an ATTRACTIVE term, so more negative
coupling => more contacts, i.e. dP/dM < 0. See map_residual for the derivation.
Getting this backwards recovers the NEGATIVE of the true coupling.)

No autodiff, no differentiating through MD. (Same result Zhang & Wolynes 2015 and
Fi-chrom use.) The chain rule down to the low-rank parameters is linear algebra:

    M = c + C Lambda C^T                       (bilinear kernel)
    dL/dLambda = C^T G C
    dL/dC      = 2 G C Lambda                  (Lambda symmetric)
    dL/dc      = sum(G)

So one optimizer step = simulate -> contact map -> subtract target -> project
onto (C, Lambda) -> Adam update. The expensive part is the simulation.

Three things make this cheap enough to be practical:

1. LOW PARAMETER COUNT. Fitting a scalar lambda (k=1) is 1 parameter against a
   whole map; k=2 is 3. Contrast Fi-chrom's ~N^2/2 free parameters against one
   noisy map -- there the gradient noise cannot be averaged away, which is the
   real reason it failed to converge in 6 days (not merely the run length).
2. WARM STARTING. Each iteration restarts from the previous iteration's final
   conformations. Lambda moves a little per step, so the ensemble is already
   nearly equilibrated under the new potential.
3. ADAPTIVE BUDGET. Short cheap runs while gradients are large; longer runs only
   when the gradient approaches its own standard error. Bounded above, so it
   cannot run away chasing the noise floor.

THE NOISE FLOOR IS REAL. Replica noise sets a limit below which no fit can go
(measured analogue: the k2 reproduction returned 2.073 against a target of
2.087). An optimizer that detects this and stops is behaving correctly; the
reported floor is also the input the identifiability analysis needs later --
parameter directions that hit the floor early are the sloppy ones.

ENVIRONMENT DISCIPLINE. The fit environment must match the simulation
environment, so the optimizer takes the SAME Simulator (hence the same
BackgroundStack) that will later be used to simulate the fitted model. It
records the background key with the result and refuses to proceed if a target
map was produced under a different one.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .model import Model, bilinear_kernel
from .simulator import (
    Simulator, SaveSpec, contact_probability,
    observed_over_expected, saddle_strength,
    N_STEPS_PRODUCTION,
)


# =====================================================================
# Adaptive simulation budget
# =====================================================================

@dataclass
class Budget:
    """How much simulation to spend on one gradient estimate.

    Adaptive, but BOUNDED -- the ceiling is ~10x the floor, so the worst case is
    known in advance and the optimizer cannot propose unbounded run lengths.

    Two triggers lengthen a run:
      (a) low signal-to-noise: |grad| < snr_target * stderr(grad). The standard
          error is estimated from the BETWEEN-REPLICA spread, which is free --
          the per-replica maps already exist.
      (b) a large parameter step: after a big move in Lambda the warm-started
          ensemble is equilibrated to the OLD potential, so a short run will not
          relax. Detected by the within-run convergence gate rather than assumed.
    """
    n_steps: int = 50_000            # production steps per replica, per iteration
    n_replicas: int = 4
    min_steps: int = 50_000
    max_steps: int = 500_000
    min_replicas: int = 2
    max_replicas: int = 20
    growth: float = 1.5              # multiplicative increase when triggered
    snr_target: float = 2.0          # want |grad| >= snr_target * stderr
    conv_target: float = 0.9         # within-run first/second-half map correlation

    def at_ceiling(self):
        return self.n_steps >= self.max_steps and self.n_replicas >= self.max_replicas

    def grow(self, reason, log=None):
        """Increase the budget. Steps first (cheaper per unit of noise reduction
        than replicas, because warm-started chains amortise equilibration), then
        replicas."""
        before = (self.n_steps, self.n_replicas)
        if self.n_steps < self.max_steps:
            self.n_steps = int(min(self.max_steps, self.n_steps * self.growth))
        elif self.n_replicas < self.max_replicas:
            self.n_replicas = int(min(self.max_replicas,
                                      np.ceil(self.n_replicas * self.growth)))
        if log is not None and (self.n_steps, self.n_replicas) != before:
            log(f"    budget up ({reason}): "
                f"{before[0]}x{before[1]} -> {self.n_steps}x{self.n_replicas}")


# =====================================================================
# Adam
# =====================================================================

class Adam:
    """Per-parameter adaptive step sizes. Used throughout rather than plain SGD:
    it is no less transparent (gradient, loss and parameter traces are all still
    visible) and it handles the different natural scales of Lambda's diagonal vs
    off-diagonal entries once k > 1. Plain gradient descent is available via
    beta1=beta2=0 for debugging."""

    def __init__(self, shape, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, grad):
        self.t += 1
        if self.b1 == 0 and self.b2 == 0:          # plain gradient descent
            return -self.lr * grad
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return -self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =====================================================================
# Objective / gradient
# =====================================================================

def map_residual(P_sim, P_exp, sep_min=3, sep_max=None):
    """Descent direction in map space, masked.

    SIGN -- read this before touching it.

    The loss is mean((P_sim - P_exp)^2), so
        dL/dM = 2 (P_sim - P_exp) * dP/dM.

    In MiChroM the coupling multiplies an ATTRACTIVE contact term: a more
    NEGATIVE coupling means stronger attraction, hence MORE contacts. So contact
    probability is a DECREASING function of the coupling, dP/dM < 0, and

        dL/dM  proportional to  (P_exp - P_sim).

    Equivalently, this is the standard maximum-entropy update
    alpha <- alpha + eta (phi_sim - phi_exp): when the simulation has too few
    contacts (phi_sim < phi_exp) the coupling must become more negative.

    Returning (P_exp - P_sim) therefore gives a quantity that can be DESCENDED.
    Using (P_sim - P_exp) drives the fit away from the answer and flips the sign
    of the recovered coupling -- observed directly in self-recovery, where a true
    lambda of -1.2 was "recovered" as +0.36 with M correlation exactly -1.000.

    (The positive factor |dP/dM| is absorbed into the learning rate.)
    """
    N = P_sim.shape[0]
    sep = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    mask = sep >= sep_min
    if sep_max is not None:
        mask &= sep <= sep_max
    G = np.where(mask, P_exp - P_sim, 0.0)
    return G, mask


def loss_standard_error(per_replica_maps, target, mask):
    """Standard error of the LOSS, from the between-replica spread.

    Free: the per-replica maps already exist. Needed because the stopping rules
    compare loss differences, and a fixed relative threshold is meaningless if it
    sits below the noise. Measured on the toy system: the loss varies ~10%
    between repeat evaluations at 4 replicas, while the default worsening
    threshold was 2% -- so four consecutive "increases" occurred by chance and
    the fit stopped at a noise-driven point 36% worse than achievable.
    """
    ls = []
    for P in per_replica_maps:
        R = np.where(mask, target - P, 0.0)
        ls.append(float((R[mask] ** 2).mean()))
    n = len(ls)
    if n < 2:
        return float("nan")
    return float(np.std(ls, ddof=1) / np.sqrt(n))


def loss_from_residual(G, mask):
    """Mean squared residual over the masked pairs. Sign-independent, so it is
    the same whether G is (P_sim - P_exp) or (P_exp - P_sim)."""
    return float((G[mask] ** 2).mean())


def project_gradient(G, model, fit_c=False, weight=None):
    """Push the map-space residual G down onto the model parameters.

    bilinear:  M = c + C Lambda C^T
        dL/dLambda = C^T (W o G) C
        dL/dC      = 2 (W o G) C Lambda     (Lambda symmetric)
        dL/dc      = sum(W o G)

    `weight` W is the CONTACT SUSCEPTIBILITY |d<f_ij>/dM_ij|, estimated as
    Var(f_ij) over frames (fluctuation-dissipation). WHY IT MATTERS:

    The full chain rule is  dL/dM = 2 (P_sim - P_exp) * dP/dM.  Treating dP/dM
    as a positive constant "absorbed into the learning rate" is harmless for a
    SINGLE parameter -- only the sign sets the direction, and the fit slides to
    the minimum along one axis. It is NOT harmless for k >= 2: the projection
    weights each pair by c_i c_j, while dP/dM is small for strongly-coupled
    pairs (they are saturated, f near 0 or 1, so they barely respond). The
    approximation therefore over-weights exactly the pairs that respond least,
    tilting the descent direction and converging to a stationary point of the
    APPROXIMATE gradient rather than the true loss minimum.

    Observed: k=1 recovered cleanly, while k=2 converged smoothly (loss trace
    descended and plateaued, scatter 1% -- not a noise artefact) to loss
    2.14e-2 when a point at 1.58e-2 existed along the line toward truth.

    weight=None reproduces the old unweighted behaviour.
    """
    if model.kernel is not bilinear_kernel:
        raise NotImplementedError(
            "gradient projection is implemented for the bilinear kernel; "
            "other kernels need their own chain rule.")
    C, Lam = model.C, model.Lam
    if weight is not None:
        W = np.asarray(weight, float)
        # normalise so the overall gradient scale (and hence a tuned learning
        # rate) is preserved; only the RELATIVE weighting matters.
        m = W.mean()
        G = G * (W / m if m > 0 else 1.0)
    gLam = C.T @ G @ C
    gLam = 0.5 * (gLam + gLam.T)               # keep Lambda symmetric
    gC = 2.0 * (G @ C @ Lam)
    gc = float(G.sum()) if fit_c else 0.0
    return gLam, gC, gc


# =====================================================================
# Result container
# =====================================================================

@dataclass
class FitResult:
    model: Model                      # BEST iterate (minimum loss), not the last
    final_model: Model = None         # last iterate, for diagnostics
    best_iter: int = -1
    history: list = field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""
    noise_floor: float = float("nan")
    background_key: dict = field(default_factory=dict)
    wall_minutes: float = 0.0

    def to_json(self, path):
        d = {
            "history": self.history,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "noise_floor": self.noise_floor,
            "background_key": self.background_key,
            "wall_minutes": self.wall_minutes,
            "C": self.model.C.tolist(),
            "Lambda": self.model.Lam.tolist(),
            "c": self.model.c,
        }
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1, default=str)

    def trace(self, key):
        return np.array([h[key] for h in self.history if key in h])


# =====================================================================
# The optimizer
# =====================================================================

class Optimizer:
    """Fit a Model's parameters to a target contact map by simulate-and-fit.

    Parameters
    ----------
    simulator : the SAME Simulator (hence BackgroundStack) that will be used to
                simulate the fitted model. Enforced, not merely recommended:
                Lambda is a residual against a fixed background, so a Lambda
                fitted in one environment is not valid in another.
    target    : (N, N) target contact probability map.
    lr_lambda, lr_C : Adam step sizes. Deliberately conservative.

                WHEN C IS TRAINABLE, SET lr_C SUBSTANTIALLY LARGER THAN
                lr_lambda (a ratio of ~3 works; see below). C and Lambda are
                coupled, and there is a trap: if C points in the wrong direction,
                the best Lambda for that wrong C is near ZERO -- switching the
                compartment term off beats getting it wrong. But the C gradient
                is  dL/dC = 2 G C Lambda,  so once Lambda ~ 0 the C gradient
                vanishes too and C can never recover. The fit collapses to "no
                structure" and self-traps there. Measured on a mock: at
                lr_C = lr_lambda the fit went to lambda = -0.03 with loss equal
                to the lambda = 0 reference; at lr_C = 3 x lr_lambda it recovered
                lambda = -1.09 (true -1.2) with |corr(C, C_true)| = 0.98. Too large a
                step overshoots and then trips the plateau test, which LOOKS like
                convergence but leaves the parameter wrong (observed on a mock
                landscape: lr 0.15 -> 10% error and an early plateau stop, lr
                0.05 -> 1% error). If the loss trace oscillates or plateaus while
                still far from the noise floor, lower the rate rather than
                raising patience.
    fit_c     : also fit the additive background level c. Off by default -- c is
                strongly degenerate with the ideal-chromosome term, so fitting
                both invites a flat direction.
    """

    def __init__(self, simulator: Simulator, target: np.ndarray,
                 lr_lambda=0.005, lr_C=0.002, fit_c=False,
                 sep_min=3, sep_max=None, budget: Budget = None,
                 out_dir=None, tag="fit", verbose=True,
                 verbose_replicas=False, quiet_sim=True):
        self.sim = simulator
        self.target = np.asarray(target, float)
        self.lr_lambda = lr_lambda
        self.lr_C = lr_C
        self.fit_c = fit_c
        self.sep_min = sep_min
        self.sep_max = sep_max
        self.budget = budget or Budget()
        self.out_dir = out_dir or os.path.join(simulator.out_dir, "fits")
        self.tag = tag
        self.verbose = verbose
        # per-replica heartbeat inside an iteration (off by default: the
        # per-iteration summary is usually enough, but a 500k x 20 iteration is
        # a long silence).
        self.verbose_replicas = verbose_replicas
        # suppress OpenMiChroM's banner + energy table, which is otherwise
        # reprinted once per replica per iteration and buries the fit log.
        if quiet_sim:
            self.sim.quiet = True
        os.makedirs(self.out_dir, exist_ok=True)

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---------- one gradient estimate ----------
    def _simulate(self, model, it, warm_start=None):
        """Run n_replicas short simulations of `model`, warm-started where
        possible. Returns (pooled map, per-replica maps, final conformations,
        convergence values)."""
        spec = SaveSpec(contact_map=True, trajectory=True, velocities=False,
                        forces=False, interval=1000, monitor=False)
        cond = f"{self.tag}_it{it:03d}"
        maps, finals, convs, varis = [], [], [], []
        t_it = time.time()
        for s in range(self.budget.n_replicas):
            init = None
            if warm_start is not None and s < len(warm_start):
                init = warm_start[s]
            # warm-started runs skip collapse/equilibration: the chain is already
            # a physical conformation, and re-equilibrating would waste the whole
            # point of warm starting.
            phases = ("production",) if init is not None else \
                     ("collapse", "equil", "production")
            self.sim.run_replica(cond, s, model=model, initial_coords=init,
                                 phases=phases,
                                 n_production=self.budget.n_steps,
                                 save_spec=spec, verbose=False, overwrite=True)
            d = self.sim.load_trajectory(cond, s)
            xyz = d["xyz"]
            Pi, Vi = contact_probability(xyz, return_var=True)
            maps.append(Pi); varis.append(Vi)
            finals.append(xyz[-1])
            md = self.sim.load_metadata(cond, s)
            convs.append(md.get("convergence", float("nan")))
            # one compact line per replica: enough to see progress on a long
            # iteration (500k x 20 is a long silence otherwise) without the
            # per-block spam that report=True produces.
            if self.verbose_replicas:
                warm = "warm" if init is not None else "cold"
                self._log(f"      rep {s+1}/{self.budget.n_replicas} ({warm})  "
                          f"conv={convs[-1]:.3f}  "
                          f"{(time.time()-t_it)/60:.1f} min elapsed")
        P = np.mean(maps, axis=0)
        self._last_susceptibility = np.mean(varis, axis=0) if varis else None
        return P, maps, finals, convs

    @staticmethod
    def _grad_stderr(per_replica_maps, target, model, mask, fit_c):
        """Standard error of the gradient, from the between-replica spread.

        Free: the per-replica maps already exist. Gives the noise scale against
        which the gradient magnitude is judged, and hence the noise floor."""
        gs = []
        for P in per_replica_maps:
            # same sign convention as map_residual (P_exp - P_sim); see there
            G = np.where(mask, target - P, 0.0)
            gLam, gC, gc = project_gradient(G, model, fit_c=fit_c)
            gs.append(np.concatenate([gLam.ravel(), gC.ravel(), [gc]]))
        gs = np.array(gs)
        n = len(gs)
        return gs.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else \
            np.full(gs.shape[1], np.nan)

    # ---------- the loop ----------
    def fit(self, model: Model, n_iter=60, patience=8, rel_tol=1e-4,
            worsen_window=4, worsen_tol=0.02, n_sigma=2.0, scale_lr=True,
            use_susceptibility=True, min_iters=12,
            gauge="unit_variance", max_conv_growths=6, collapse_tol=1e-2,
            history_path=None):
        """Run the simulate-and-fit loop.

        Stops when any of:
          - the gradient is inside its own standard error at the budget ceiling
            (the NOISE FLOOR -- a legitimate result, not a failure),
          - the loss RISES for `worsen_window` consecutive iterations and is
            `worsen_tol` above the best seen (overshoot; lower the learning rate),
          - the loss stops improving for `patience` iterations,
          - n_iter is reached.

        Returns the BEST iterate in `.model` and the last in `.final_model`.
        """
        t0 = time.time()
        model = Model(model.C.copy(), model.Lam.copy(), c=model.c,
                      kernel=model.kernel, C_trainable=model.C_trainable,
                      Lam_trainable=model.Lam_trainable, names=model.names)

        if model.C_trainable:
            model.gauge_fix(mode=gauge)      # start in the fixed gauge

        # A learning rate tuned on a single parameter is too large once Lambda
        # has several entries of differing scales. Scale it down with the
        # parameter count unless the caller has opted out.
        n_lam = model.k * (model.k + 1) // 2
        lr_L = self.lr_lambda / np.sqrt(n_lam) if scale_lr else self.lr_lambda
        if scale_lr and n_lam > 1:
            self._log(f"  lr_lambda scaled {self.lr_lambda:.4g} -> {lr_L:.4g} "
                      f"for {n_lam} Lambda parameters")
        adam_L = Adam(model.Lam.shape, lr=lr_L)
        adam_C = Adam(model.C.shape, lr=self.lr_C)
        adam_c = Adam((), lr=self.lr_lambda)

        res = FitResult(model=model,
                        background_key=self.sim.background.key())
        best, best_it, warm = np.inf, -1, None
        # snapshot of the best-so-far parameters. Returning the LAST iterate is
        # wrong for a noisy objective: the optimizer can walk past the minimum
        # and sit there while the plateau test counts down. Observed directly --
        # a k=1 fit reached lambda ~ -1.25 at the loss minimum, then drifted to
        # -1.53 (28% error) before stopping, with the loss rising the whole way.
        best_params = (model.Lam.copy(), model.C.copy(), model.c)
        conv_growths = 0

        for it in range(n_iter):
            P, maps, finals, convs = self._simulate(model, it, warm_start=warm)
            warm = finals                       # warm-start the next iteration

            G, mask = map_residual(P, self.target, self.sep_min, self.sep_max)
            loss = loss_from_residual(G, mask)
            loss_se = loss_standard_error(maps, self.target, mask)
            W = getattr(self, "_last_susceptibility", None) if use_susceptibility else None
            gLam, gC, gc = project_gradient(G, model, fit_c=self.fit_c, weight=W)

            se = self._grad_stderr(maps, self.target, model, mask, self.fit_c)
            gvec = np.concatenate([gLam.ravel(), gC.ravel(), [gc]])
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.abs(gvec) / se
            snr_max = float(np.nanmax(snr)) if np.isfinite(snr).any() else np.nan
            med_conv = float(np.nanmedian(convs))

            rec = {
                "iter": it, "loss": loss,
                "grad_norm": float(np.linalg.norm(gvec)),
                "snr_max": snr_max,
                "loss_se": loss_se,
                # The loss contains a REPLICA-NOISE FLOOR of order sigma^2/n, so
                # loss values are only comparable between runs with the SAME
                # n_replicas and n_steps. Recorded here to make that checkable.
                "loss_sampling": [int(self.budget.n_replicas),
                                  int(self.budget.n_steps)],
                "n_steps": self.budget.n_steps,
                "n_replicas": self.budget.n_replicas,
                "median_convergence": med_conv,
                "Lambda": model.Lam.tolist(),
                "c": model.c,
                # C diagnostics: with the gauge fixed, C_std should stay ~1.
                # A growing C_absmax with flat C_std means the coordinate is
                # developing extreme outliers -> expect convergence to fall.
                "lam_absmax": float(np.abs(model.Lam).max()),
                "C_std": float(model.C.std()),
                "C_absmax": float(np.abs(model.C).max()),
                "M_min": float(model.coupling_matrix().min()),
                "M_max": float(model.coupling_matrix().max()),
            }
            res.history.append(rec)
            if history_path:                 # incremental: survives interruption
                with open(history_path, "w") as fh:
                    json.dump(res.history, fh, indent=1, default=str)
            self._log(f"  it {it:3d}  loss {loss:.4e}+-{loss_se:.1e}  "
                      f"|g| {rec['grad_norm']:.3e}  "
                      f"snr {snr_max:.2f}  conv {med_conv:.3f}  "
                      f"|C|max {rec['C_absmax']:.2f}  "
                      f"M[{rec['M_min']:+.2f},{rec['M_max']:+.2f}]  "
                      f"budget {self.budget.n_steps}x{self.budget.n_replicas}")

            # ---- budget adaptation ----
            # (a) gradient approaching its own noise
            if np.isfinite(snr_max) and snr_max < self.budget.snr_target:
                if self.budget.at_ceiling():
                    res.converged = True
                    res.stop_reason = ("noise floor: gradient within its standard "
                                       "error at maximum budget")
                    res.noise_floor = loss
                    self._log(f"  STOP -- {res.stop_reason}")
                    break
                self.budget.grow("low snr", self._log)
            # (b) warm-started ensemble not relaxed under the new potential
            elif np.isfinite(med_conv) and med_conv < self.budget.conv_target:
                conv_growths += 1
                if conv_growths > max_conv_growths and self.budget.at_ceiling():
                    # Growing the budget assumes longer runs will equilibrate.
                    # If the POTENTIAL itself does not relax (e.g. gauge drift
                    # has produced extreme couplings, or the coordinate has gone
                    # glassy), more sampling never helps and the optimizer burns
                    # its whole allowance. Stop and say so.
                    res.stop_reason = (
                        f"ensembles not equilibrating (median convergence "
                        f"{med_conv:.3f} < {self.budget.conv_target}) at the "
                        f"budget ceiling -- the potential is not relaxing, not "
                        f"under-sampled. Check |C| and the coupling range.")
                    res.converged = False
                    res.noise_floor = best
                    self._log(f"  STOP -- {res.stop_reason}")
                    break
                self.budget.grow("low within-run convergence", self._log)

            # ---- parameter update ----
            if model.Lam_trainable:
                model.Lam = model.Lam + adam_L.step(gLam)
                model.Lam = 0.5 * (model.Lam + model.Lam.T)
            if model.C_trainable:
                model.C = model.C + adam_C.step(gC)
                # REMOVE THE GAUGE FREEDOM EVERY ITERATION. C -> CA,
                # Lambda -> A^-1 Lambda A^-T leaves M unchanged, so the loss
                # exerts no force along it and the optimizer drifts: C inflates,
                # couplings become extreme, the polymer goes glassy and stops
                # equilibrating. Without this, a free-C run drove the simulation
                # budget to its ceiling while within-run convergence sat at 0.82.
                model.gauge_fix(mode=gauge)
            if self.fit_c:
                model.c = model.c + float(adam_c.step(np.array(gc)))

            # ---- Lambda-collapse check (free C only) ----
            if model.C_trainable:
                lam_scale = float(np.abs(model.Lam).max())
                if lam_scale < collapse_tol:
                    res.stop_reason = (
                        f"Lambda collapsed to ~0 (max|Lambda| = {lam_scale:.2e}). "
                        f"The fit has switched the compartment term OFF, and "
                        f"because dL/dC = 2 G C Lambda the C gradient has "
                        f"vanished with it -- C can no longer learn. Raise lr_C "
                        f"relative to lr_lambda (try 3x) and restart.")
                    res.converged = False
                    res.noise_floor = best
                    self._log(f"  STOP -- {res.stop_reason}")
                    break

            # ---- plateau / divergence check (NOISE-AWARE) ----
            # Thresholds are set from the measured loss noise, not a fixed
            # fraction. A constant threshold below the noise level makes the
            # stopping rules fire at random.
            # Take the LARGER of the relative and noise-based thresholds.
            # Using n_sigma*loss_se ALONE is a mistake: loss_se is the standard
            # error of the mean over replicas, which is far smaller than the
            # iteration-to-iteration scatter (different conformations, warm-start
            # drift). Substituting it made the worsening rule fire on the first
            # few noise-level upticks and return the STARTING model unchanged.
            if np.isfinite(loss_se) and loss_se > 0:
                improve_thresh = max(rel_tol * best, n_sigma * loss_se)
                worsen_thresh = max(worsen_tol * best, n_sigma * loss_se)
            else:
                improve_thresh = rel_tol * best
                worsen_thresh = worsen_tol * best

            # NOTE the isfinite guard. `best` starts at inf, so
            # improve_thresh = max(rel_tol*best, ...) is inf and
            # `best - improve_thresh` is inf - inf = nan; every comparison
            # against nan is False, so WITHOUT this guard the first iteration
            # never registers, best_params is never set, and the fit returns the
            # STARTING model while reporting "best iterate: iter -1, loss inf"
            # -- even though the loss descended monotonically throughout.
            if (not np.isfinite(best)) or (loss < best - improve_thresh):
                best, best_it = loss, it
                best_params = (model.Lam.copy(), model.C.copy(), model.c)
            else:
                # sustained WORSENING is a stronger signal than mere flatness:
                # it means the step size is carrying the parameters past the
                # minimum. Stop sooner than the plain plateau rule would.
                recent = [h["loss"] for h in res.history[-worsen_window:]]
                if (it >= min_iters
                        and len(recent) == worsen_window
                        and all(np.diff(recent) > 0)
                        and loss > best + worsen_thresh):
                    res.stop_reason = (
                        f"loss rising for {worsen_window} iterations and "
                        f"{n_sigma:.0f}+ sigma above the best (iter {best_it}); "
                        f"loss noise +-{loss_se:.2e}")
                    res.converged = True
                    res.noise_floor = best
                    self._log(f"  STOP -- {res.stop_reason}")
                    break
                if it >= min_iters and it - best_it >= patience:
                    res.stop_reason = f"loss plateau ({patience} iterations)"
                    res.converged = True
                    res.noise_floor = best
                    self._log(f"  STOP -- {res.stop_reason}")
                    break
        else:
            res.stop_reason = f"reached n_iter={n_iter}"

        # return the BEST iterate; keep the last one for diagnostics
        res.final_model = model
        res.best_iter = best_it
        bLam, bC, bc = best_params
        res.model = Model(bC, bLam, c=bc, kernel=model.kernel,
                          C_trainable=model.C_trainable,
                          Lam_trainable=model.Lam_trainable, names=model.names)
        self._log(f"  best iterate: iter {best_it}, loss {best:.4e}")
        res.wall_minutes = (time.time() - t0) / 60.0
        res.to_json(os.path.join(self.out_dir, f"{self.tag}.json"))
        self._log(f"  done in {res.wall_minutes:.1f} min -- {res.stop_reason}")
        return res


# =====================================================================
# Self-recovery: the honesty gate
# =====================================================================

def make_synthetic_target(simulator: Simulator, true_model: Model,
                          n_replicas=20, n_production=N_STEPS_PRODUCTION,
                          cond="truth", verbose=True):
    """Simulate a KNOWN theta and treat the resulting map as 'experimental'.

    Why synthetic rather than a real chr10 sub-region: a sub-region's measured
    map includes contacts with the rest of the chromosome and the rest of the
    nucleus, which an isolated simulation does not reproduce. Fitting it would
    silently absorb that boundary mismatch into Lambda -- the same fit-vs-sim
    environment mismatch the retraining rule warns about. A synthetic target is
    generated in exactly the environment it will be fitted in, so ground truth
    is both known AND self-consistent.
    """
    spec = SaveSpec(contact_map=True, trajectory=True, velocities=False,
                    interval=1000, monitor=False)
    for s in range(n_replicas):
        simulator.run_replica(cond, s, model=true_model,
                              n_production=n_production,
                              save_spec=spec, verbose=verbose)
    return simulator.pool(cond, n_replicas)


def replica_noise_ceiling(simulator: Simulator, model: Model, n_replicas=20,
                          n_production=N_STEPS_PRODUCTION, cond="ceiling",
                          verbose=True):
    """Correlation between two INDEPENDENT ensembles of the SAME theta.

    This is what perfect recovery looks like numerically. The recovery
    correlation ceiling is NOT 1.0, and comparing a fit against 1.0 rather than
    against this number will make a correct fit look like a failure.
    """
    spec = SaveSpec(contact_map=True, trajectory=False, monitor=False)
    for s in range(2 * n_replicas):
        simulator.run_replica(cond, s, model=model, n_production=n_production,
                              save_spec=spec, verbose=verbose)
    import glob
    files = sorted(glob.glob(os.path.join(simulator.out_dir, "maps",
                                          f"{cond}_rep*.npy")),
                   key=lambda p: int(p.split("rep")[-1].split(".")[0]))
    files = [f for f in files if ".conv." not in f]
    A = np.mean([np.load(f).astype(float) for f in files[:n_replicas]], axis=0)
    B = np.mean([np.load(f).astype(float) for f in files[n_replicas:2 * n_replicas]],
                axis=0)
    iu = np.triu_indices(A.shape[0], 3)
    return float(np.corrcoef(A[iu], B[iu])[0, 1]), A, B


def recovery_report(true_model: Model, fitted_model: Model):
    """Compare recovered to true parameters GAUGE-INVARIANTLY.

    Raw C and Lambda differ by a gauge transformation even on perfect recovery
    (C -> MC, Lambda -> M^-T Lambda M^-1 leaves M = C Lambda C^T unchanged), so
    comparing them directly is meaningless. Compare instead:

      primary    : the coupling matrix M -- gauge-invariant by construction, and
                   the thing the simulation actually feels
      diagnostic : M's spectrum (rotation invariant)
      structural : Procrustes alignment of C. The residual says how well C is
                   recovered up to rotation; the ROTATION MAGNITUDE is itself the
                   identifiability diagnostic -- near-identity means the
                   parameterization is pinned down, a large rotation means a flat
                   direction was found.

    Failure type is diagnosable: M recovered but C/Lambda not = benign gauge
    freedom; M not recovered = real non-identifiability or optimizer failure.
    """
    Mt = true_model.coupling_matrix()
    Mf = fitted_model.coupling_matrix()
    rel = float(np.linalg.norm(Mf - Mt) / np.linalg.norm(Mt))
    iu = np.triu_indices(Mt.shape[0], 1)
    corr = float(np.corrcoef(Mt[iu], Mf[iu])[0, 1])

    # Scale, measured separately from shape. CRITICAL: correlation is BLIND to
    # scale, and for k=1 it is degenerate -- M = c + lambda*C C^T means the
    # off-diagonal structure is proportional to C C^T for ANY lambda, so the
    # correlation is identically 1.000 even when lambda is wrong by a factor of
    # eight (verified). Never report correlation as the headline recovery metric;
    # M_relative_error and the scale ratio below carry the magnitude information.
    a, b = Mt[iu] - Mt[iu].mean(), Mf[iu] - Mf[iu].mean()
    scale_ratio = float((b @ a) / (a @ a)) if a @ a > 0 else float("nan")

    wt = np.linalg.eigvalsh(Mt); wt = wt[np.argsort(-np.abs(wt))]
    wf = np.linalg.eigvalsh(Mf); wf = wf[np.argsort(-np.abs(wf))]
    k = min(6, len(wt))

    out = {
        "M_relative_error": rel,          # PRIMARY: sensitive to shape AND scale
        "M_scale_ratio": scale_ratio,     # 1.0 = correct magnitude
        "M_correlation": corr,            # shape only; DEGENERATE for k=1
        "spectrum_true": wt[:k].tolist(),
        "spectrum_fitted": wf[:k].tolist(),
    }

    # Procrustes on C (only meaningful when shapes match)
    if true_model.C.shape == fitted_model.C.shape:
        A, B = true_model.C, fitted_model.C
        U, S, Vt = np.linalg.svd(B.T @ A)
        R = U @ Vt                                   # optimal rotation B -> A
        resid = float(np.linalg.norm(B @ R - A) / max(np.linalg.norm(A), 1e-12))
        rot = float(np.linalg.norm(R - np.eye(R.shape[0])))
        out["procrustes_residual"] = resid
        out["rotation_magnitude"] = rot
        out["rotation_matrix"] = R.tolist()
    return out


def self_recovery(simulator: Simulator, true_model: Model, init_model: Model,
                  n_iter=60, target_replicas=20,
                  target_production=N_STEPS_PRODUCTION,
                  budget: Budget = None, tag="selfrec", verbose=True,
                  compute_ceiling=True, **fit_kw):
    """The honesty gate: invent theta, simulate it, treat that map as
    experimental, fit theta back, check recovery.

    Returns (FitResult, report dict). The report includes the replica-noise
    ceiling so recovery quality is judged against what is actually achievable
    rather than against 1.0.
    """
    if verbose:
        print("[1/3] generating synthetic target from known theta ...")
    target = make_synthetic_target(simulator, true_model,
                                   n_replicas=target_replicas,
                                   n_production=target_production,
                                   cond=f"{tag}_truth", verbose=verbose)

    ceiling = float("nan")
    if compute_ceiling:
        if verbose:
            print("[2/3] measuring replica-noise ceiling ...")
        ceiling, _, _ = replica_noise_ceiling(
            simulator, true_model, n_replicas=target_replicas // 2,
            n_production=target_production, cond=f"{tag}_ceil", verbose=verbose)

    if verbose:
        print("[3/3] fitting ...")
    opt = Optimizer(simulator, target, budget=budget, tag=tag,
                    verbose=verbose, **fit_kw)
    res = opt.fit(init_model, n_iter=n_iter)

    report = recovery_report(true_model, res.model)
    report["replica_noise_ceiling"] = ceiling
    report["final_loss"] = res.history[-1]["loss"] if res.history else None
    report["stop_reason"] = res.stop_reason
    report["iterations"] = len(res.history)

    if verbose:
        print("\n--- recovery report ---")
        print(f"  M relative error   : {report['M_relative_error']:.4f}   <- PRIMARY")
        print(f"  M scale ratio      : {report['M_scale_ratio']:.4f}   (1.0 = right magnitude)")
        print(f"  M correlation      : {report['M_correlation']:.4f}   (shape only; "
              f"degenerate for k=1)")
        print(f"  replica ceiling    : {ceiling:.4f}   <- perfect recovery target")
        if "procrustes_residual" in report:
            print(f"  Procrustes residual: {report['procrustes_residual']:.4f}")
            print(f"  rotation magnitude : {report['rotation_magnitude']:.4f}"
                  f"   (near 0 = pinned; large = flat direction)")
        print(f"  true spectrum      : {np.round(report['spectrum_true'], 4)}")
        print(f"  fitted spectrum    : {np.round(report['spectrum_fitted'], 4)}")
    return res, report


# =====================================================================
# Synthetic C generators (for self-recovery across the C-modes)
# =====================================================================

def synthetic_types(n_beads, types=("A1", "A2", "B1", "B2", "B3"),
                    mean_run=12, seed=0):
    """A plausible per-bead type sequence: contiguous domains with realistic
    run lengths and no singletons (chr10's Rao track has 221 runs, zero
    singletons, mean run 12.3 beads)."""
    rng = np.random.default_rng(seed)
    seq = []
    while len(seq) < n_beads:
        t = types[rng.integers(len(types))]
        run = max(2, int(rng.lognormal(np.log(mean_run), 0.5)))
        seq.extend([t] * run)
    return seq[:n_beads]


def synthetic_coordinate(n_beads, k=1, n_domains=20, seed=0, smooth=3):
    """A smooth continuous per-bead coordinate: piecewise-constant domain values,
    lightly smoothed, standardised to unit variance per column."""
    rng = np.random.default_rng(seed)
    C = np.zeros((n_beads, k))
    for col in range(k):
        edges = np.sort(rng.choice(np.arange(1, n_beads), n_domains - 1,
                                   replace=False))
        vals = rng.normal(0, 1, n_domains)
        seg = np.split(np.arange(n_beads), edges)
        x = np.zeros(n_beads)
        for s, v in zip(seg, vals):
            x[s] = v
        if smooth > 1:
            kern = np.ones(smooth) / smooth
            x = np.convolve(x, kern, mode="same")
        C[:, col] = (x - x.mean()) / (x.std() + 1e-12)
    return C