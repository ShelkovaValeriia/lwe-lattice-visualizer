from typing import Tuple

import numpy as np


def find_shortest_vector(
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Finds the shortest non-zero lattice vector by brute force.

    This is an educational SVP implementation for small examples.
    """
    non_zero_mask = np.any(coefficient_vectors != 0, axis=1)

    non_zero_coefficients = coefficient_vectors[non_zero_mask]
    non_zero_points = lattice_points[non_zero_mask]

    norms = np.linalg.norm(non_zero_points, axis=1)
    min_index = int(np.argmin(norms))

    return (
        non_zero_coefficients[min_index],
        non_zero_points[min_index],
        float(norms[min_index]),
    )


def find_closest_point(
    target: np.ndarray,
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solves CVP approximately by brute force inside the selected range.
    """
    distances = np.linalg.norm(lattice_points - target, axis=1)
    min_index = int(np.argmin(distances))

    return (
        coefficient_vectors[min_index],
        lattice_points[min_index],
        float(distances[min_index]),
    )


def gauss_reduce_2d(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs simple Gaussian lattice basis reduction in 2D.

    The new basis B_reduced defines the same lattice:
    B_reduced = B * U,
    where U is an integer unimodular matrix with det(U) = ±1.
    """
    if B.shape != (2, 2):
        raise ValueError("Gauss reduction is implemented only for 2D bases.")

    B_reduced = B.copy().astype(float)
    U = np.eye(2, dtype=int)

    changed = True

    while changed:
        changed = False

        b1 = B_reduced[:, 0]
        b2 = B_reduced[:, 1]

        if np.dot(b2, b2) < np.dot(b1, b1):
            B_reduced[:, [0, 1]] = B_reduced[:, [1, 0]]
            U[:, [0, 1]] = U[:, [1, 0]]
            changed = True
            continue

        b1 = B_reduced[:, 0]
        b2 = B_reduced[:, 1]

        mu = int(round(np.dot(b1, b2) / np.dot(b1, b1)))

        if mu != 0:
            B_reduced[:, 1] = B_reduced[:, 1] - mu * B_reduced[:, 0]
            U[:, 1] = U[:, 1] - mu * U[:, 0]
            changed = True

    return B_reduced, U