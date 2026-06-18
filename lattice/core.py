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


def generate_lattice_points(B: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates lattice points of the form v = Bz,
    where z is an integer vector with coordinates in [-radius, radius].

    Basis vectors are stored as columns of B.
    """
    n = B.shape[0]

    coefficient_vectors = np.array(
        list(itertools.product(range(-radius, radius + 1), repeat=n)),
        dtype=int,
    )

    lattice_points = coefficient_vectors @ B.T

    return coefficient_vectors, lattice_points


def lattice_determinant(B: np.ndarray) -> float:
    """
    Returns |det(B)|.
    """
    return abs(float(np.linalg.det(B)))