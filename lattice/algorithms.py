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


def basis_score(B: np.ndarray) -> float:
    """
    Computes a simple quality score for a basis.

    Smaller score means a more convenient basis:
    - shorter vectors,
    - more orthogonal vectors.
    """
    n = B.shape[1]
    vectors = [B[:, i] for i in range(n)]
    norms = [np.linalg.norm(v) for v in vectors]

    if any(norm == 0 for norm in norms):
        return float("inf")

    length_part = sum(norms)

    orthogonality_part = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            cosine = abs(float(np.dot(vectors[i], vectors[j])) / (norms[i] * norms[j]))
            orthogonality_part += cosine * max(norms[i], norms[j])

    return float(length_part + orthogonality_part)


def generate_unimodular_matrices_2d(limit: int = 3) -> List[np.ndarray]:
    """
    Generates 2D integer unimodular matrices U with det(U) = ±1.

    If B is a basis, then B_new = B * U is another basis
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


def generate_unimodular_matrices_by_column_operations(
    dimension: int,
    coefficient_limit: int = 3,
    max_steps: int = 2,
) -> List[np.ndarray]:
    """
    Generates unimodular matrices using elementary column operations.

    Operations:
    - swap two columns,
    - change sign of a column,
    - add k times one column to another column.

    These operations always preserve det(U) = ±1.
    """
    identity = np.eye(dimension, dtype=int)

    matrices = [identity]
    seen = {tuple(identity.flatten())}

    current_level = [identity]

    for _ in range(max_steps):
        next_level = []

        for U in current_level:
            candidates = []

            # Swap columns.
            for i in range(dimension):
                for j in range(i + 1, dimension):
                    V = U.copy()
                    V[:, [i, j]] = V[:, [j, i]]
                    candidates.append(V)

            # Change sign of a column.
            for i in range(dimension):
                V = U.copy()
                V[:, i] = -V[:, i]
                candidates.append(V)

            # Add k times column j to column i.
            for i in range(dimension):
                for j in range(dimension):
                    if i == j:
                        continue

                    for k in range(-coefficient_limit, coefficient_limit + 1):
                        if k == 0:
                            continue

                        V = U.copy()
                        V[:, i] = V[:, i] + k * V[:, j]
                        candidates.append(V)

            for V in candidates:
                key = tuple(V.flatten())

                if key in seen:
                    continue

                det = round(float(np.linalg.det(V)))

                if abs(det) != 1:
                    continue

                seen.add(key)
                matrices.append(V)
                next_level.append(V)

        current_level = next_level

    return matrices


def generate_basis_candidates(
    B: np.ndarray,
    limit: int = 3,
    max_candidates: int = 8,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generates convenient and inconvenient bases for the same lattice.

    Each candidate has:
    - basis: B_new = B * U
    - U: unimodular matrix
    - score: quality score

    For 2D, it enumerates small unimodular matrices directly.
    For 3D, it generates unimodular matrices using elementary column operations.
    """
    n = B.shape[0]

    if B.shape[0] != B.shape[1]:
        return [], []

    if n == 2:
        unimodular_matrices = generate_unimodular_matrices_2d(limit=limit)
    elif n == 3:
        unimodular_matrices = generate_unimodular_matrices_by_column_operations(
            dimension=3,
            coefficient_limit=limit,
            max_steps=2,
        )
    else:
        return [], []

    candidates = []
    seen = set()

    for U in unimodular_matrices:
        new_basis = B @ U
        key = tuple(np.round(new_basis.flatten(), 8))

        if key in seen:
            continue

        seen.add(key)

        norms = [np.linalg.norm(new_basis[:, i]) for i in range(n)]
        score = basis_score(new_basis)

        candidates.append(
            {
                "basis": new_basis,
                "U": U,
                "score": score,
                "max_vector_length": max(norms),
                "sum_vector_lengths": sum(norms),
            }
        )

    convenient = sorted(candidates, key=lambda item: item["score"])
    inconvenient = sorted(candidates, key=lambda item: item["score"], reverse=True)

    return convenient[:max_candidates], inconvenient[:max_candidates]


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