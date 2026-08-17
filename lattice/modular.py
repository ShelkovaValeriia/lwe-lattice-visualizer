import itertools
from typing import Optional, Tuple, Union

import numpy as np


NumericInput = Union[int, np.ndarray]
NumericOutput = Union[int, np.ndarray]


def _validate_modulus(modulus: int) -> None:
    """Validate a modulus used by modular arithmetic helpers."""
    if not isinstance(modulus, (int, np.integer)):
        raise ValueError("Modulus q must be an integer.")

    if int(modulus) < 2:
        raise ValueError("Modulus q must be at least 2.")


def is_prime(value: int) -> bool:
    """
    Return True if value is a prime number.

    This implementation is sufficient for the small educational moduli
    used by the visualizer.
    """
    if not isinstance(value, (int, np.integer)):
        return False

    value = int(value)

    if value < 2:
        return False

    if value == 2:
        return True

    if value % 2 == 0:
        return False

    divisor = 3

    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2

    return True


def previous_prime(value: int) -> Optional[int]:
    """
    Return the largest prime strictly smaller than value.

    Returns None when there is no smaller positive prime.
    """
    candidate = int(value) - 1

    while candidate >= 2:
        if is_prime(candidate):
            return candidate
        candidate -= 1

    return None


def next_prime(value: int) -> int:
    """Return the smallest prime strictly greater than value."""
    candidate = max(2, int(value) + 1)

    while not is_prime(candidate):
        candidate += 1

    return candidate


def nearest_primes(value: int) -> Tuple[Optional[int], int]:
    """
    Return the neighboring prime numbers around value.

    Examples:
        nearest_primes(8) -> (7, 11)
        nearest_primes(12) -> (11, 13)
    """
    return previous_prime(value), next_prime(value)


def recommended_prime(value: int) -> Optional[int]:
    """
    Return the uniquely closest neighboring prime.

    If both neighboring primes are equally far away, return None so the
    interface can present both choices without preferring either one.
    """
    lower, upper = nearest_primes(value)

    if lower is None:
        return upper

    lower_distance = value - lower
    upper_distance = upper - value

    if lower_distance < upper_distance:
        return lower

    if upper_distance < lower_distance:
        return upper

    return None


def mod_reduce(values: NumericInput, modulus: int) -> NumericOutput:
    """
    Reduce values to the standard residue representation [0, q - 1].

    Examples for q = 5:
        8  -> 3
        -1 -> 4
    """
    _validate_modulus(modulus)
    modulus = int(modulus)

    reduced = np.mod(np.asarray(values), modulus).astype(int)

    if np.isscalar(values):
        return int(reduced)

    return reduced


def center_mod(values: NumericInput, modulus: int) -> NumericOutput:
    """
    Convert values to a centered residue representation modulo q.

    For an odd modulus q, the result lies in:
        [-(q // 2), ..., q // 2]

    Example for q = 5:
        0, 1, 2, 3, 4 -> 0, 1, 2, -2, -1
    """
    _validate_modulus(modulus)
    modulus = int(modulus)

    reduced = np.asarray(mod_reduce(values, modulus))

    centered = (
        (reduced + modulus // 2) % modulus
    ) - modulus // 2

    centered = centered.astype(int)

    if np.isscalar(values):
        return int(centered)

    return centered


def generate_coefficient_vectors(
    dimension: int,
    coefficient_limit: int,
) -> np.ndarray:
    """
    Generate all integer coefficient vectors z in [-r, r]^dimension.
    """
    if dimension < 1:
        raise ValueError("Dimension must be at least 1.")

    if coefficient_limit < 0:
        raise ValueError("Coefficient range must be non-negative.")

    return np.array(
        list(
            itertools.product(
                range(-coefficient_limit, coefficient_limit + 1),
                repeat=dimension,
            )
        ),
        dtype=int,
    )


def generate_modular_lattice_points(
    B: np.ndarray,
    modulus: int,
    coefficient_limit: int,
    centered: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate points x = Bz (mod q).

    Returns:
        coefficient_vectors:
            Integer vectors z used to generate the points.
        standard_points:
            Residues represented in [0, q - 1].
        display_points:
            Either centered residues or standard residues, depending on
            the selected visualization mode.
    """
    _validate_modulus(modulus)

    matrix = np.asarray(B)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Basis matrix B must be square.")

    if not np.allclose(matrix, np.round(matrix)):
        raise ValueError(
            "Basis matrix B must contain only integers in modular mode."
        )

    matrix = np.round(matrix).astype(int)

    coefficient_vectors = generate_coefficient_vectors(
        dimension=matrix.shape[0],
        coefficient_limit=coefficient_limit,
    )

    raw_points = coefficient_vectors @ matrix.T
    standard_points = np.asarray(
        mod_reduce(raw_points, modulus),
        dtype=int,
    )

    if centered:
        display_points = np.asarray(
            center_mod(standard_points, modulus),
            dtype=int,
        )
    else:
        display_points = standard_points.copy()

    return coefficient_vectors, standard_points, display_points


def summarize_unique_modular_points(
    coefficient_vectors: np.ndarray,
    standard_points: np.ndarray,
    display_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse repeated modular residues for plotting.

    Different coefficient vectors can map to the same residue after the
    modulo operation. The function keeps one representative coefficient
    vector and records how many mappings produced each displayed point.
    """
    if not (
        len(coefficient_vectors)
        == len(standard_points)
        == len(display_points)
    ):
        raise ValueError(
            "Coefficient vectors and modular point arrays must have equal length."
        )

    unique_display_points, first_indices, multiplicities = np.unique(
        display_points,
        axis=0,
        return_index=True,
        return_counts=True,
    )

    unique_standard_points = standard_points[first_indices]
    representative_coefficients = coefficient_vectors[first_indices]

    return (
        unique_display_points,
        unique_standard_points,
        representative_coefficients,
        multiplicities,
    )


def _integer_determinant(matrix: np.ndarray) -> int:
    """Compute an exact determinant for the small integer matrices we use."""
    matrix = np.asarray(matrix, dtype=int)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    size = matrix.shape[0]

    if size == 1:
        return int(matrix[0, 0])

    if size == 2:
        return int(
            matrix[0, 0] * matrix[1, 1]
            - matrix[0, 1] * matrix[1, 0]
        )

    determinant = 0

    for column in range(size):
        minor = np.delete(
            np.delete(matrix, 0, axis=0),
            column,
            axis=1,
        )
        determinant += (
            (-1) ** column
            * int(matrix[0, column])
            * _integer_determinant(minor)
        )

    return int(determinant)


def determinant_mod_q(B: np.ndarray, modulus: int) -> int:
    """Return det(B) reduced modulo q."""
    _validate_modulus(modulus)

    matrix = np.asarray(B)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Basis matrix B must be square.")

    if not np.allclose(matrix, np.round(matrix)):
        raise ValueError(
            "Basis matrix B must contain only integers in modular mode."
        )

    matrix = np.round(matrix).astype(int)

    return _integer_determinant(matrix) % int(modulus)


def is_invertible_mod_q(B: np.ndarray, modulus: int) -> bool:
    """
    Return True when B is invertible modulo q.

    In the application q is restricted to primes, so a non-zero determinant
    modulo q is sufficient for invertibility over Z_q.
    """
    return determinant_mod_q(B, modulus) != 0