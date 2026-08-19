import itertools

import numpy as np

from lattice.modular import is_prime


def _validate_integer_matrix(A: np.ndarray) -> np.ndarray:
    """Return A as an integer matrix after validating its shape and entries."""
    matrix = np.asarray(A)

    if matrix.ndim != 2:
        raise ValueError("Constraint matrix A must be two-dimensional.")

    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            "Constraint matrix A must have at least one row and one column."
        )

    if not np.allclose(matrix, np.round(matrix)):
        raise ValueError(
            "Constraint matrix A must contain only integer entries."
        )

    return np.round(matrix).astype(int)


def _validate_prime_modulus(modulus: int) -> int:
    """Validate and return the prime modulus used by the q-ary lattice mode."""
    if not isinstance(modulus, (int, np.integer)):
        raise ValueError("Prime modulus q must be an integer.")

    modulus = int(modulus)

    if not is_prime(modulus):
        raise ValueError("Q-ary lattice mode requires q to be prime.")

    return modulus


def rank_mod_q(A: np.ndarray, modulus: int) -> int:
    """
    Compute the rank of A over the finite field Z_q for prime q.

    The implementation uses Gaussian elimination with modular inverses.
    """
    matrix = _validate_integer_matrix(A)
    modulus = _validate_prime_modulus(modulus)
    reduced = np.mod(matrix, modulus).astype(int)

    row_count, column_count = reduced.shape
    rank = 0
    pivot_column = 0

    while rank < row_count and pivot_column < column_count:
        pivot_row = None

        for candidate_row in range(rank, row_count):
            if reduced[candidate_row, pivot_column] % modulus != 0:
                pivot_row = candidate_row
                break

        if pivot_row is None:
            pivot_column += 1
            continue

        if pivot_row != rank:
            reduced[[rank, pivot_row]] = reduced[[pivot_row, rank]]

        pivot_value = int(reduced[rank, pivot_column])
        pivot_inverse = pow(pivot_value, -1, modulus)

        reduced[rank] = (
            reduced[rank] * pivot_inverse
        ) % modulus

        for row_index in range(row_count):
            if row_index == rank:
                continue

            factor = int(reduced[row_index, pivot_column])

            if factor != 0:
                reduced[row_index] = (
                    reduced[row_index]
                    - factor * reduced[rank]
                ) % modulus

        rank += 1
        pivot_column += 1

    return rank


def generate_qary_lattice_points(
    A: np.ndarray,
    modulus: int,
    coordinate_limit: int,
) -> np.ndarray:
    """
    Generate the visible part of the q-ary lattice

        Lambda_q^perp(A) = {x in Z^m : A x = 0 (mod q)}.

    Only integer points whose coordinates lie in [-L, L] are returned.
    The coordinate limit controls the displayed window, not the lattice itself.
    """
    matrix = _validate_integer_matrix(A)
    modulus = _validate_prime_modulus(modulus)

    if not isinstance(coordinate_limit, (int, np.integer)):
        raise ValueError("Coordinate limit L must be an integer.")

    coordinate_limit = int(coordinate_limit)

    if coordinate_limit < 0:
        raise ValueError(
            "Coordinate limit L must be non-negative."
        )

    dimension = matrix.shape[1]

    candidates = np.array(
        list(
            itertools.product(
                range(
                    -coordinate_limit,
                    coordinate_limit + 1,
                ),
                repeat=dimension,
            )
        ),
        dtype=int,
    )

    if candidates.size == 0:
        return np.empty(
            (0, dimension),
            dtype=int,
        )

    modular_values = np.mod(
        candidates @ matrix.T,
        modulus,
    )

    valid_mask = np.all(
        modular_values == 0,
        axis=1,
    )

    return candidates[valid_mask]


def qary_lattice_index(
    A: np.ndarray,
    modulus: int,
) -> int:
    """
    Return the index [Z^m : Lambda_q^perp(A)] for prime q.

    If r is the rank of A over Z_q, then the image of x -> A x has q^r
    elements. Therefore the kernel lift Lambda_q^perp(A) has index q^r.
    """
    modulus = _validate_prime_modulus(modulus)
    rank = rank_mod_q(A, modulus)

    return modulus**rank


def qary_lattice_determinant(
    A: np.ndarray,
    modulus: int,
) -> int:
    """
    Return det(Lambda_q^perp(A)) for the q-ary sublattice of Z^m.

    Since Z^m has determinant 1, the determinant of a full-rank sublattice
    equals its index in Z^m.
    """
    return qary_lattice_index(
        A,
        modulus,
    )