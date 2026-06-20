import ast
import itertools
from typing import Tuple

import numpy as np


EPS = 1e-9


def parse_matrix(text: str) -> np.ndarray:
    """
    Parses a basis matrix from text input.

    Example:
    [[2, 1],
     [0, 1]]
    """
    try:
        data = ast.literal_eval(text)
        matrix = np.array(data, dtype=float)
    except Exception as exc:
        raise ValueError(
            "Matrix must be written as a valid Python list, "
            "for example [[2, 1], [0, 1]]."
        ) from exc

    if matrix.ndim != 2:
        raise ValueError("Basis matrix must be two-dimensional.")

    rows, cols = matrix.shape

    if rows != cols:
        raise ValueError("Basis matrix must be square, for example 2x2 or 3x3.")

    det = np.linalg.det(matrix)

    if abs(det) < EPS:
        raise ValueError(
            "Basis matrix is singular. Its determinant is 0, "
            "so it does not define a full-rank lattice."
        )

    return matrix


def parse_vector(text: str, dimension: int) -> np.ndarray:
    """
    Parses target vector from text input.

    Example:
    [2.3, 1.7]
    """
    try:
        data = ast.literal_eval(text)
        vector = np.array(data, dtype=float)
    except Exception as exc:
        raise ValueError(
            "Target point must be written as a valid Python list, "
            "for example [2.3, 1.7]."
        ) from exc

    if vector.ndim != 1:
        raise ValueError("Target point must be a vector, for example [2.3, 1.7].")

    if len(vector) != dimension:
        raise ValueError(f"Target point must have dimension {dimension}.")

    return vector


def lattice_determinant(B: np.ndarray) -> float:
    """
    Returns |det(B)|.
    """
    return abs(float(np.linalg.det(B)))


def estimate_coefficient_search_radius(B: np.ndarray, cube_limit: int) -> int:
    """
    Estimates how large the coefficient search range for z must be
    so that all lattice points inside the coordinate cube [-L, L]^n
    can be found.

    Since z = B^{-1} x and ||x||_inf <= L,
    we use ||z||_inf <= ||B^{-1}||_inf * L.
    """
    B_inv = np.linalg.inv(B)
    bound = np.linalg.norm(B_inv, ord=np.inf) * cube_limit
    return max(1, int(np.ceil(bound)) + 1)


def generate_lattice_points_in_cube(
    B: np.ndarray,
    cube_limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates all lattice points x = Bz that lie inside the coordinate cube:
        [-cube_limit, cube_limit]^n

    Internally:
    1. estimate a sufficient coefficient search radius for z,
    2. generate candidate z vectors,
    3. keep only those points whose coordinates are inside the cube.
    """
    n = B.shape[0]
    search_radius = estimate_coefficient_search_radius(B, cube_limit)

    coefficient_vectors = np.array(
        list(itertools.product(range(-search_radius, search_radius + 1), repeat=n)),
        dtype=int,
    )

    lattice_points = coefficient_vectors @ B.T

    mask = np.all(np.abs(lattice_points) <= cube_limit + EPS, axis=1)

    return coefficient_vectors[mask], lattice_points[mask]