from typing import Optional

import numpy as np
import streamlit as st


BASIS_MODES = [
    "Input basis",
    "Convenient basis",
    "Inconvenient basis",
]


def choose_basis(
    B: np.ndarray,
    convenient_bases: list[dict],
    inconvenient_bases: list[dict],
    label: str,
    mode_key: str,
    default_mode_index: int,
) -> tuple[np.ndarray, str, Optional[np.ndarray]]:
    """
    Lets the user choose only the basis type.

    The app automatically selects the best generated basis of that type.
    """
    if B.shape not in [(2, 2), (3, 3)]:
        st.selectbox(
            label,
            options=["Input basis"],
            key=mode_key,
            disabled=True,
        )
        return B, "Input basis", None

    basis_mode = st.selectbox(
        label,
        options=BASIS_MODES,
        index=default_mode_index,
        key=mode_key,
        help=(
            "Input basis is the matrix from the sidebar. "
            "Convenient and inconvenient bases are generated from it by unimodular "
            "transformations, so they show the same lattice with different basis vectors."
        ),
    )

    if basis_mode == "Input basis":
        return B, "Input basis", None

    if basis_mode == "Convenient basis":
        if not convenient_bases:
            st.warning("No convenient basis was generated for this input.")
            return B, "Input basis", None

        selected = convenient_bases[0]
        return selected["basis"], "Convenient basis", selected["U"]

    if basis_mode == "Inconvenient basis":
        if not inconvenient_bases:
            st.warning("No inconvenient basis was generated for this input.")
            return B, "Input basis", None

        selected = inconvenient_bases[0]
        return selected["basis"], "Inconvenient basis", selected["U"]

    return B, "Input basis", None


def show_normal_visualization_controls(
    B_input: np.ndarray,
    convenient_bases: list[dict],
    inconvenient_bases: list[dict],
) -> tuple[bool, np.ndarray, str, Optional[np.ndarray]]:
    """
    Shows visualization controls in normal mode.
    """
    header_col, switch_col = st.columns([2, 1])

    with header_col:
        st.subheader("Visualization")

    with switch_col:
        comparison_mode = st.checkbox(
            "Comparison mode",
            value=False,
            key="comparison_mode",
            help=(
                "Show two selected bases side by side. "
                "The lattice points stay the same."
            ),
        )

    B_displayed, basis_mode, unimodular_matrix = choose_basis(
        B=B_input,
        convenient_bases=convenient_bases,
        inconvenient_bases=inconvenient_bases,
        label="Displayed basis",
        mode_key="single_basis_mode",
        default_mode_index=1,
    )

    return comparison_mode, B_displayed, basis_mode, unimodular_matrix


def show_comparison_visualization_controls(
    B_input: np.ndarray,
    convenient_bases: list[dict],
    inconvenient_bases: list[dict],
) -> tuple[
    bool,
    np.ndarray,
    str,
    Optional[np.ndarray],
    np.ndarray,
    str,
    Optional[np.ndarray],
]:
    """
    Shows comparison controls on full page width.

    There are two independent selectors:
    - one for the left graph;
    - one for the right graph.
    """
    header_col, switch_col = st.columns([2, 1])

    with header_col:
        st.subheader("Visualization")

    with switch_col:
        comparison_mode = st.checkbox(
            "Comparison mode",
            value=True,
            key="comparison_mode",
            help=(
                "Show two selected bases side by side. "
                "The lattice points stay the same."
            ),
        )

    left_selector_col, right_selector_col = st.columns(2)

    with left_selector_col:
        B_left, left_label, left_unimodular_matrix = choose_basis(
            B=B_input,
            convenient_bases=convenient_bases,
            inconvenient_bases=inconvenient_bases,
            label="Left graph basis",
            mode_key="left_basis_mode",
            default_mode_index=0,
        )

    with right_selector_col:
        B_right, right_label, right_unimodular_matrix = choose_basis(
            B=B_input,
            convenient_bases=convenient_bases,
            inconvenient_bases=inconvenient_bases,
            label="Right graph basis",
            mode_key="right_basis_mode",
            default_mode_index=1,
        )

    return (
        comparison_mode,
        B_left,
        left_label,
        left_unimodular_matrix,
        B_right,
        right_label,
        right_unimodular_matrix,
    )