from typing import Optional

import numpy as np
import streamlit as st

from lattice.utils import format_array


def get_fundamental_domain_label(n: int) -> str:
    """
    Returns a readable name for the determinant interpretation.
    """
    if n == 2:
        return "Fundamental area"

    if n == 3:
        return "Fundamental volume"

    return "Fundamental domain volume"


def show_basis_vectors(B: np.ndarray) -> None:
    """
    Shows basis vectors as columns of the basis matrix.
    """
    st.write("### Basis vectors")

    for i in range(B.shape[1]):
        st.write(f"b{i + 1} = `{format_array(B[:, i])}`")


def show_same_lattice_indicator(unimodular_matrix: Optional[np.ndarray]) -> None:
    """
    Shows whether the displayed basis defines the same lattice.
    """
    if unimodular_matrix is None:
        st.info("This is the basis entered in the input panel.")
        return

    det_u = round(float(np.linalg.det(unimodular_matrix)))

    if abs(det_u) == 1:
        st.success(f"Same lattice: det(U) = {det_u}")
    else:
        st.error(f"Warning: det(U) = {det_u}, so the lattice may be different.")


def show_results(
    B_displayed: np.ndarray,
    det_value: float,
    shortest_coefficients: np.ndarray,
    shortest_vector: np.ndarray,
    shortest_length: float,
    use_target: bool,
    target: Optional[np.ndarray],
    closest_coefficients: Optional[np.ndarray],
    closest_point: Optional[np.ndarray],
    closest_distance: Optional[float],
    basis_mode: str,
    unimodular_matrix: Optional[np.ndarray],
) -> None:
    """
    Shows compact numerical results.
    """
    n = B_displayed.shape[0]

    st.subheader("Results")
    st.metric(get_fundamental_domain_label(n), round(det_value, 6))

    st.write("### Displayed basis")
    st.write(basis_mode)
    show_same_lattice_indicator(unimodular_matrix)
    show_basis_vectors(B_displayed)

    with st.expander("Show basis matrix"):
        st.code(str(B_displayed.tolist()), language="text")

    if unimodular_matrix is not None:
        with st.expander("Show same-lattice transformation"):
            st.write("The displayed basis is generated as:")
            st.code("B_new = B · U", language="text")
            st.write("Unimodular matrix U:")
            st.code(str(unimodular_matrix.tolist()), language="text")

    st.write("### Shortest vector / SVP")
    st.write(f"Coefficient vector z: `{shortest_coefficients.tolist()}`")
    st.write(f"Shortest vector Bz: `{format_array(shortest_vector)}`")
    st.write(f"Length: `{round(shortest_length, 6)}`")

    if use_target and target is not None:
        st.write("### Closest vector / CVP")
        st.write(f"Target point t: `{format_array(target)}`")
        st.write(f"Coefficient vector z: `{closest_coefficients.tolist()}`")
        st.write(f"Closest lattice point Bz: `{format_array(closest_point)}`")
        st.write(f"Distance: `{round(closest_distance, 6)}`")


def show_explanation_box(n: int) -> None:
    """
    Shows a short educational explanation of the visualization.
    """
    with st.expander("What is shown?"):
        st.write(
            "The app constructs lattice points in the form `Bz`, "
            "where `B` is the basis matrix and `z` is an integer vector."
        )

        st.write(
            "Basis vectors are interpreted as the columns of the matrix `B`."
        )

        st.write(
            "The determinant `|det(B)|` represents the size of the fundamental domain. "
            "In 2D it is the area of the fundamental parallelogram. "
            "In 3D it is the volume of the fundamental parallelepiped."
        )

        if n in [2, 3]:
            st.write(
                "Convenient and inconvenient bases are generated as `B_new = B · U`, "
                "where `U` is an integer unimodular matrix with `det(U) = ±1`. "
                "That changes the basis vectors, but not the lattice itself."
            )

        st.write(
            "SVP and CVP are computed by brute force among the lattice points "
            "inside the selected coordinate cube. This is suitable for small "
            "educational examples, not for high-dimensional cryptographic lattices."
        )