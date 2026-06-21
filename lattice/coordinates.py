from typing import Optional

import numpy as np


def coefficients_for_displayed_basis(
    coefficients: np.ndarray,
    unimodular_matrix: Optional[np.ndarray],
) -> np.ndarray:
    """
    Converts coefficient vectors from the input basis to the displayed basis.

    If B_displayed = B_input · U, then every lattice point can be written as:
        B_input · z_input = B_displayed · z_displayed

    Therefore:
        z_input = U · z_displayed
        z_displayed = U^{-1} · z_input

    This is especially important in comparison mode, where both plots use the
    same lattice points, but each plot may display a different basis.
    """
    if unimodular_matrix is None:
        return coefficients

    converted = np.linalg.solve(unimodular_matrix, coefficients.T).T
    rounded = np.rint(converted).astype(int)

    return rounded