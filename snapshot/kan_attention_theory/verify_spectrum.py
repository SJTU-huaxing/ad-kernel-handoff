"""Reproduce the numerical examples in the KAN / attention theory discussion.

Requires only NumPy. These are checks of idealized mathematical examples,
not measurements of an LLM and not evidence of KAN training performance.

Run: python /root/autodl-tmp/kan_attention_theory/verify_spectrum.py
"""

import json
import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


def gaussian_spectrum_check(a, nodes=100, modes=6):
    """T f(x) = E_y[exp(a*x*y) f(y)], x,y independently N(0,1)."""
    x, weights = hermegauss(nodes)
    weights /= np.sqrt(2 * np.pi)
    matrix = np.exp(a * np.outer(x, x)) * np.sqrt(np.outer(weights, weights))
    observed = np.linalg.eigvalsh(matrix)[::-1][:modes]
    s = np.sqrt(1 - 4 * a * a)
    t = 2 * a / (1 + s)
    predicted = np.sqrt(1 + t * t) * t ** np.arange(modes)
    return {
        "a": a,
        "predicted": predicted.tolist(),
        "quadrature": observed.tolist(),
        "max_absolute_difference": float(np.max(np.abs(observed - predicted))),
    }


def isotropic_relative_mse(d, m):
    """Exact population rank-m floor for exp(q.k/sqrt(d)), q,k ~ N(0,I)."""
    if d <= 4 or m < 1:
        raise ValueError("Require d > 4 (Hilbert-Schmidt) and m >= 1.")
    a = 1 / np.sqrt(d)
    t = 2 * a / (1 + np.sqrt(1 - 4 * a * a))
    z = t * t
    retained_energy = 0.0
    degree = 0
    remaining = m
    while remaining:
        multiplicity = math.comb(d + degree - 1, degree)
        retained_modes = min(remaining, multiplicity)
        retained_energy += retained_modes * (1 - z) ** d * z ** degree
        remaining -= retained_modes
        degree += 1
    return float(1 - retained_energy)


def nonnegative_constraint_example():
    """A positive exponential kernel whose unique best rank-2 fit is negative.

    For this finite measure, nonnegative matrices of rank <= 2 form a closed
    set. The unique unrestricted minimizer has a negative entry, so even that
    larger closed class has a strictly larger minimum. Nonnegative-factor
    rank-2 approximations consequently also have a strictly larger minimum.
    """
    x = np.array([-3.0, -1.0, 1.0, 3.0])
    probabilities = np.array([0.01, 0.49, 0.49, 0.01])
    kernel = np.exp(0.2 * np.outer(x, x))
    scale = np.sqrt(np.outer(probabilities, probabilities))
    weighted = scale * kernel
    eigenvalues, eigenvectors = np.linalg.eigh(weighted)
    weighted_rank_two = (
        eigenvectors[:, -2:] * eigenvalues[-2:]
    ) @ eigenvectors[:, -2:].T
    rank_two = weighted_rank_two / scale
    return {
        "support": x.tolist(),
        "probabilities": probabilities.tolist(),
        "eigenvalues_descending": eigenvalues[::-1].tolist(),
        "unrestricted_rank_two_mse": float(np.sum(eigenvalues[:-2] ** 2)),
        "minimum_entry_of_rank_two_kernel": float(np.min(rank_two)),
    }


def factor_perturbation_check(trials=100):
    """Check the excess-risk inequality on finite dimensional HS operators.

    With F*=U_m sqrt(S_m), G*=V_m sqrt(S_m), perturb both factors.
    If e_q=||F-F*||_F and e_k=||G-G*||_F, then
      ||A-FG^T||_F^2 <= tail
        + (sqrt(sigma_1)*(e_q+e_k)+e_q*e_k)^2
        + 2*sigma_{m+1}*e_q*e_k.
    Orthogonality removes first-order cross terms with the spectral residual.
    """
    rng = np.random.default_rng(20260905)
    worst_violation = -float("inf")
    for _ in range(trials):
        matrix = rng.normal(size=(9, 7))
        u, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
        m = 3
        f = u[:, :m] * np.sqrt(sigma[:m])
        g = vt[:m].T * np.sqrt(sigma[:m])
        df = rng.normal(size=f.shape) * 0.03
        dg = rng.normal(size=g.shape) * 0.03
        eq, ek = np.linalg.norm(df), np.linalg.norm(dg)
        actual = np.linalg.norm(matrix - (f + df) @ (g + dg).T) ** 2
        upper = (
            np.sum(sigma[m:] ** 2)
            + (np.sqrt(sigma[0]) * (eq + ek) + eq * ek) ** 2
            + 2 * sigma[m] * eq * ek
        )
        worst_violation = max(worst_violation, actual - upper)
    return {"trials": trials, "max_actual_minus_upper_bound": worst_violation}


if __name__ == "__main__":
    result = {
        "scope": "Independent Gaussian/discrete toy distributions; not LLM measurements",
        "one_dimensional_spectrum": [gaussian_spectrum_check(a) for a in (0.1, 0.2, 0.4)],
        "isotropic_examples": [
            {"d": d, "m": m, "relative_kernel_mse_floor": isotropic_relative_mse(d, m)}
            for d, m in ((64, 128), (128, 256), (128, 8385), (128, 366145))
        ],
        "nonnegative_example": nonnegative_constraint_example(),
        "factor_perturbation": factor_perturbation_check(),
    }
    print(json.dumps(result, indent=2))
