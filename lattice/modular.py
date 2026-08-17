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