"""
model.py  --  Part A of the chromatin potential machinery: the Model object.

The keystone abstraction. A Model holds:
    C       : (N, k) coordinates     -- each bead's state vector
    Lam     : (k, k) coupling         -- the coupling operator
    c       : scalar background level  -- the "removed mean" added back
    kernel  : callable (C, Lam) -> (N, N) coupling matrix

and emits the (N, N) coupling matrix M that the simulator consumes:

    bilinear:  M[i,j] = c + (C @ Lam @ C.T)[i,j]

Every model in the project is a choice of (C-type, Lam-structure, kernel):
    - MiChroM 5-state   : one-hot C (k=5), full 5x5 Lam, bilinear
    - rank-2 (k2)       : one-hot C (k=5), rank-2 Lam,   bilinear
    - continuous PC1    : scalar C  (k=1), [lambda_1],   bilinear
    - blind reduced-rank: free C (rank k), free Lam,     bilinear
    - Fi-chrom          : one-hot per bead (k=N), NxN Lam, bilinear (trivial)

This module is PURE NUMPY -- no simulation, no fitting, no GPU. It is testable
on CPU. Simulation lives in the simulator wrapper (Part C); fitting lives in the
optimizer. Feature extraction for the surrogate has stub methods reserved here
so downstream code has a stable interface.

Conventions (decompose/reconstruct) are copied verbatim from the validated
michrom_decomposition notebook so results are guaranteed identical.
"""

from __future__ import annotations
import numpy as np


# =====================================================================
# Decomposition helpers -- verbatim conventions from michrom_decomposition.ipynb
# =====================================================================

def decompose(A, p, center=True, weight=True):
    """Eigendecompose the (optionally centered, optionally abundance-weighted)
    symmetric matrix A. Returns (w, V, W, c):
        w : eigenvalues sorted by |.| descending
        V : eigenvectors (columns), same order
        W : per-type weight vector (sqrt abundance, or ones)
        c : removed level (scalar)
    """
    if center:
        c = (p @ A @ p) if weight else A.mean()
    else:
        c = 0.0
    D = A - c
    W = np.sqrt(p) if weight else np.ones(len(p))
    M = np.diag(W) @ D @ np.diag(W)
    w, V = np.linalg.eigh(M)                 # eigh valid: A symmetric
    idx = np.argsort(-np.abs(w))
    return w[idx], V[:, idx], W, c


def reconstruct(w, V, W, c, k):
    """Simulate-able matrix from the top-k modes: un-weight, add the level back
    (native units). This is the rank-k reconstruction of A."""
    Dk = np.diag(1 / W) @ (V[:, :k] @ np.diag(w[:k]) @ V[:, :k].T) @ np.diag(1 / W)
    return Dk + c


def factorize(A, p, k, center=True, weight=False):
    """Factor A into (C, Lam, c) such that  c + C @ Lam @ C.T  is the rank-k
    reconstruction of A. Default center=True, weight=False matches the k2 arm.

    C   = un-weighted top-k eigvecs scaled by sqrt|w|,  shape (n_types, k)
    Lam = diag(sign(w_k)),                              shape (k, k)
    c   = removed level
    Then  C @ Lam @ C.T = sum_m sign(w_m) * |w_m| * (v_m v_m^T) = reconstruction - c.
    """
    w, V, W, c = decompose(A, p, center=center, weight=weight)
    Vk = np.diag(1 / W) @ V[:, :k]                 # un-weight the eigenvectors
    C = Vk * np.sqrt(np.abs(w[:k]))[None, :]        # (n_types, k)
    Lam = np.diag(np.sign(w[:k]))                   # (k, k)
    return C, Lam, c


# =====================================================================
# Kernels -- pluggable. bilinear now; RBF / neural slot in later.
# =====================================================================

def bilinear_kernel(C, Lam):
    """M_struct[i,j] = (C @ Lam @ C.T)[i,j]. The background level c is added by
    the Model, not here, so kernels only produce the structured part."""
    return C @ Lam @ C.T


def rbf_kernel(C, Lam):
    """Placeholder for the distance/similarity kernel (Aim 3 kernel comparison).
    Not implemented yet -- reserved so the interface is stable."""
    raise NotImplementedError("RBF kernel is a later Aim-3 experiment.")


# =====================================================================
# The Model object
# =====================================================================

class Model:
    """(C, Lam, kernel) -> N x N coupling matrix.

    Attributes
    ----------
    C            : (N, k) float           coordinates
    Lam          : (k, k) float           coupling
    c            : float                   background level (added to every entry)
    kernel       : callable(C, Lam)->(N,N)
    C_trainable  : bool                    optimizer may vary C
    Lam_trainable: bool                    optimizer may vary Lam
    names        : list[str] or None       per-row labels (bead/type names)

    The C-mode (fixed / free / free_rank_k) is expressed purely by the
    trainable flags plus how C was initialised -- the coupling computation is
    identical in all modes.
    """

    def __init__(self, C, Lam, c=0.0, kernel=bilinear_kernel,
                 C_trainable=False, Lam_trainable=True, names=None):
        self.C = np.asarray(C, float)
        self.Lam = np.asarray(Lam, float)
        self.c = float(c)
        self.kernel = kernel
        self.C_trainable = C_trainable
        self.Lam_trainable = Lam_trainable
        self.names = names
        self._check()

    def _check(self):
        assert self.C.ndim == 2, "C must be (N, k)"
        N, k = self.C.shape
        assert self.Lam.shape == (k, k), f"Lam must be ({k},{k}), got {self.Lam.shape}"
        assert np.allclose(self.Lam, self.Lam.T, atol=1e-10), "Lam must be symmetric"
        if self.names is not None:
            assert len(self.names) == N, "names length must equal N"

    # ---- dimensions ----
    @property
    def N(self):
        return self.C.shape[0]

    @property
    def k(self):
        return self.C.shape[1]

    # ---- the core contract ----
    def coupling_matrix(self):
        """Return the (N, N) coupling matrix M the simulator consumes."""
        M = self.c + self.kernel(self.C, self.Lam)
        # enforce exact symmetry (guards against float asymmetry in kernels)
        return 0.5 * (M + M.T)

    # ---- constructors: build the standard arms ----
    @classmethod
    def from_type_matrix(cls, A_full, labels_full, abundances=None,
                         real_types=(0, 1, 2, 3, 4), k=None,
                         center=True, weight=False, **kw):
        """Build a Model from a MiChroM-style type-type matrix.

        A_full       : (T, T) full type matrix (e.g. the 7x7 with A1..NA)
        labels_full  : list of T type names
        abundances   : dict name->count or None (uniform). Only affects the
                       weighted decomposition; k2 uses weight=False so abundances
                       are irrelevant there.
        real_types   : indices of the types to keep (default: A1,A2,B1,B2,B3)
        k            : rank to keep (None -> full rank of the kept block)

        C is one-hot over the kept types (so M reduces to A[type_i, type_j]);
        Lam is the k-truncated coupling; c is the removed level.
        Returns a Model whose coupling_matrix() over one-hot C equals the
        rank-k reconstruction of the kept block.
        """
        A_full = np.asarray(A_full, float)
        real = list(real_types)
        L = [labels_full[i] for i in real]
        A = A_full[np.ix_(real, real)]
        n = len(real)
        if abundances is None:
            p = np.ones(n) / n
        else:
            p = np.array([abundances.get(t, 0) for t in L], float)
            p = p / p.sum()
        if k is None:
            k = n
        C_types, Lam, c = factorize(A, p, k, center=center, weight=weight)
        # one-hot C over the kept types: M = onehot @ (C_types Lam C_types^T) @ onehot^T
        # But we want a per-TYPE model here (N = n types). Callers expand to beads
        # via a label sequence using .expand_to_beads(). So C = C_types directly.
        return cls(C_types, Lam, c=c, names=L, C_trainable=False,
                   Lam_trainable=True, **kw)

    @classmethod
    def from_coordinate(cls, coord, Lam, c=0.0, names=None, **kw):
        """Continuous model: coord is (N, k) per-bead coordinates (e.g. PC1),
        Lam is (k, k). This is the 'fixed-C' arm."""
        coord = np.asarray(coord, float)
        if coord.ndim == 1:
            coord = coord[:, None]
        return cls(coord, np.atleast_2d(Lam), c=c, names=names,
                   C_trainable=False, Lam_trainable=True, **kw)

    @classmethod
    def free(cls, N, k, seed=0, scale=1.0, c=0.0, smooth=5, lam0=-0.5, **kw):
        """Blind reduced-rank init: free C (rank k) and free Lambda, both learned
        ('reduced-rank Fi-chrom with no biology in').

        Two choices here matter for whether the fit is well behaved:

        * C is SMOOTHED along the chain, not iid noise. An iid C with an
          attractive Lambda is a random heteropolymer -- the textbook glass
          former -- which equilibrates badly and can trap the optimizer in a
          non-relaxing state from the first iteration. Smoothing gives
          domain-like structure of the kind real coordinates have, without
          encoding any actual biology.
        * Lambda starts ATTRACTIVE (negative) and weak. The previous default
          (+1, repulsive) had to cross zero to reach any sensible potential.

        C is gauge-fixed to unit variance on construction.
        """
        rng = np.random.default_rng(seed)
        C = rng.normal(0, 1.0, (N, k))
        if smooth and smooth > 1:
            kern = np.ones(smooth) / smooth
            C = np.apply_along_axis(
                lambda v: np.convolve(v, kern, mode="same"), 0, C)
        C = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-12) * scale
        Lam = np.eye(k) * lam0
        m = cls(C, Lam, c=c, C_trainable=True, Lam_trainable=True, **kw)
        m.gauge_fix()
        return m

    # ---- bead expansion (types -> per-bead) ----
    def expand_to_beads(self, label_sequence):
        """Given a per-bead sequence of type names, return a new per-bead Model
        whose C has one row per bead (the type's coordinate). Used to turn a
        per-TYPE model (from_type_matrix) into the per-BEAD potential the
        simulator needs.
        """
        assert self.names is not None, "need type names to expand"
        index = {name: i for i, name in enumerate(self.names)}
        rows = [index[l] for l in label_sequence]
        C_bead = self.C[rows, :]
        return Model(C_bead, self.Lam, c=self.c, kernel=self.kernel,
                     C_trainable=self.C_trainable,
                     Lam_trainable=self.Lam_trainable,
                     names=list(label_sequence))

    # ---- simulator bridge: the .ff format ----
    def write_ff(self, path, fmt="%.6E", bead_names=None):
        """Write the per-bead N x N coupling matrix in the .ff format the
        simulator/OpenMiChroM custom-types path consumes: a header row of bead
        names, then the N x N matrix.

        This matches the format of k2_potential.ff and the continuous notebook.
        For a per-TYPE model, pass bead_names or expand_to_beads first.
        """
        M = self.coupling_matrix()
        N = M.shape[0]
        if bead_names is None:
            bead_names = self.names if self.names is not None else \
                [f"t{i:05d}" for i in range(N)]
        with open(path, "w") as f:
            f.write(",".join(bead_names) + "\n")
            for i in range(N):
                f.write(",".join(fmt % M[i, j] for j in range(N)) + "\n")

    # ---- surrogate feature stubs (reserved interface; filled later) ----
    def demixing_contrast(self, isA, isB):
        """A-A + B-B - 2*A-B on the coupling matrix. isA, isB are boolean
        per-bead masks. A candidate theta-feature for the surrogate."""
        M = self.coupling_matrix()
        aa = M[np.ix_(isA, isA)].mean()
        bb = M[np.ix_(isB, isB)].mean()
        ab = M[np.ix_(isA, isB)].mean()
        return aa + bb - 2 * ab

    def spectrum(self):
        """Eigenvalues of Lam (|.| descending). A candidate theta-feature."""
        w = np.linalg.eigvalsh(self.Lam)
        return w[np.argsort(-np.abs(w))]

    def gauge_fix(self, mode="unit_variance"):
        """Remove the gauge freedom in (C, Lambda), IN PLACE.

        The bilinear coupling M = c + C Lambda C^T is invariant under
            C -> C A,   Lambda -> A^-1 Lambda A^-T
        for any invertible A. For k=1 this is the scalar freedom
            C -> sC,    lambda -> lambda / s^2.

        MUST be applied every iteration when C is trainable. Otherwise the
        optimizer drifts along this exactly-flat direction: C inflates while
        Lambda shrinks, M is unchanged so the loss gives no resistance, but the
        individual bead couplings become extreme and the polymer turns glassy
        and stops equilibrating. Observed directly in a free-C run -- within-run
        convergence fell to 0.82 and stayed there while the simulation budget
        was driven to its ceiling trying to fix it by sampling harder.

        mode:
          'unit_variance' : each column of C standardised to unit variance,
                            the scale absorbed into Lambda. Cheap, and it is the
                            convention the optimized-tier plan specifies.
          'orthonormal'   : columns orthonormalised (QR), the mixing absorbed
                            into Lambda. For k >= 2 this also removes the
                            rotational part of the freedom, not just the scale.
        Returns the applied A (for diagnostics).
        """
        C = self.C
        if mode == "unit_variance":
            s = C.std(axis=0)
            s = np.where(s > 1e-12, s, 1.0)
            A = np.diag(1.0 / s)                 # C -> C A
        elif mode == "orthonormal":
            Q, R = np.linalg.qr(C)
            # C = Q R  ->  choose A = R^-1 so that C A = Q
            A = np.linalg.inv(R + np.eye(R.shape[0]) * 1e-12)
        else:
            raise ValueError(f"unknown gauge mode {mode!r}")
        Ainv = np.linalg.inv(A)
        self.C = C @ A
        self.Lam = Ainv @ self.Lam @ Ainv.T
        self.Lam = 0.5 * (self.Lam + self.Lam.T)
        return A


# =====================================================================
# Self-test: reproduce the k2 result (pure linear algebra, no simulation)
# =====================================================================

# MiChroM's shipped 7x7 type matrix (A1,A2,B1,B2,B3,B4,NA), from the notebook.
_L_FULL = ["A1", "A2", "B1", "B2", "B3", "B4", "NA"]
_A_FULL = np.array([
    [-0.268, -0.275, -0.263, -0.259, -0.267, -0.267, -0.226],
    [-0.275, -0.299, -0.287, -0.281, -0.301, -0.301, -0.245],
    [-0.263, -0.287, -0.342, -0.322, -0.337, -0.337, -0.210],
    [-0.259, -0.281, -0.322, -0.330, -0.329, -0.329, -0.283],
    [-0.267, -0.301, -0.337, -0.329, -0.341, -0.341, -0.349],
    [-0.267, -0.301, -0.337, -0.329, -0.341, -0.341, -0.349],
    [-0.226, -0.245, -0.210, -0.283, -0.349, -0.349, -0.256],
])


def _selftest(verbose=True):
    """Reproduce the k2 reconstruction exactly via the Model object.

    Acceptance: Model.from_type_matrix(k=2) over one-hot C must equal the
    notebook's A_k2 (centered, unweighted rank-2 reconstruction) to machine
    precision.
    """
    real = [0, 1, 2, 3, 4]
    A = _A_FULL[np.ix_(real, real)]
    p = np.ones(5) / 5

    # reference: the notebook's exact computation
    w, V, W, c = decompose(A, p, center=True, weight=False)
    A_k2_ref = reconstruct(w, V, W, c, k=2)

    # via the Model object
    m = Model.from_type_matrix(_A_FULL, _L_FULL, real_types=real, k=2,
                               center=True, weight=False)
    M = m.coupling_matrix()          # should equal A_k2_ref (5x5, per-type)

    err = np.linalg.norm(M - A_k2_ref) / np.linalg.norm(A_k2_ref)
    maxerr = np.abs(M - A_k2_ref).max()

    # also confirm rank-2 reconstruction error vs true A matches the known 1.7%
    rel_vs_A = np.linalg.norm(A_k2_ref - A) / np.linalg.norm(A)

    if verbose:
        print("k2 self-test")
        print(f"  Model vs notebook A_k2 : rel {err:.2e}, max {maxerr:.2e}")
        print(f"  rank-2 recon err vs A  : {rel_vs_A:.2%}  (expect ~1.7%)")
        print(f"  Lam (should be diag +/-1):\n{m.Lam}")
        print(f"  removed level c        : {m.c:+.4f}")
        print("  PASS" if err < 1e-10 else "  FAIL")

    assert err < 1e-10, f"k2 reproduction failed: rel err {err:.2e}"
    return m


if __name__ == "__main__":
    _selftest()
