from typing import Dict, List, Tuple

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


def basis_score_2d(B: np.ndarray) -> float:
    """
    Computes a simple quality score for a 2D basis.

    Smaller score means a more convenient basis:
    - shorter vectors,
    - more orthogonal vectors.
    """
    b1 = B[:, 0]
    b2 = B[:, 1]

    norm1 = np.linalg.norm(b1)
    norm2 = np.linalg.norm(b2)

    if norm1 == 0 or norm2 == 0:
        return float("inf")

    cosine = abs(float(np.dot(b1, b2)) / (norm1 * norm2))

    return float(norm1 + norm2 + cosine * max(norm1, norm2))


def generate_unimodular_matrices_2d(limit: int = 3) -> List[np.ndarray]:
    """
    Generates 2D integer unimodular matrices U with det(U) = ±1.

    If B is a basis, then B' = B * U is another basis
    of the same lattice.
    """
    matrices = []

    for a in range(-limit, limit + 1):
        for b in range(-limit, limit + 1):
            for c in range(-limit, limit + 1):
                for d in range(-limit, limit + 1):
                    U = np.array([[a, b], [c, d]], dtype=int)
                    det = round(float(np.linalg.det(U)))

                    if abs(det) == 1:
                        matrices.append(U)

    return matrices


def generate_basis_candidates_2d(
    B: np.ndarray,
    limit: int = 3,
    max_candidates: int = 8,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generates convenient and inconvenient bases for the same 2D lattice.

    Each candidate has:
    - basis: B' = B * U
    - U: unimodular matrix
    - score: quality score
    """
    if B.shape != (2, 2):
        return [], []

    candidates = []
    seen = set()

    for U in generate_unimodular_matrices_2d(limit):
        new_basis = B @ U

        key = tuple(np.round(new_basis.flatten(), 8))

        if key in seen:
            continue

        seen.add(key)

        score = basis_score_2d(new_basis)

        candidates.append(
            {
                "basis": new_basis,
                "U": U,
                "score": score,
                "max_vector_length": max(
                    np.linalg.norm(new_basis[:, 0]),
                    np.linalg.norm(new_basis[:, 1]),
                ),
            }
        )

    convenient = sorted(candidates, key=lambda item: item["score"])
    inconvenient = sorted(candidates, key=lambda item: item["score"], reverse=True)

    return convenient[:max_candidates], inconvenient[:max_candidates]