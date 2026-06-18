from typing import Optional

import numpy as np
import streamlit as st

from lattice.algorithms import (
    find_closest_point,
    find_shortest_vector,
    gauss_reduce_2d,
)
from lattice.core import (
    generate_lattice_points,
    lattice_determinant,
    parse_matrix,
    parse_vector,
)
from lattice.plotting_2d import create_2d_plot
from lattice.utils import format_array


def show_results(
    B: np.ndarray,
    n: int,
    total_points: int,
    det_value: float,
    shortest_coefficients: np.ndarray,
    shortest_vector: np.ndarray,
    shortest_length: float,
    use_target: bool,
    target: Optional[np.ndarray],
    closest_coefficients: Optional[np.ndarray],
    closest_point: Optional[np.ndarray],
    closest_distance: Optional[float],
    reduced_basis: Optional[np.ndarray],
    unimodular_matrix: Optional[np.ndarray],
) -> None:
    """
    Shows numerical results in the left column.
    """
    st.subheader("Results")

    st.metric("Dimension", n)
    st.metric("Number of lattice points", total_points)
    st.metric("|det(B)|", round(det_value, 6))

    st.write("### Basis matrix")
    st.code(format_array(B), language="text")

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

    if reduced_basis is not None and unimodular_matrix is not None:
        st.write("### Reduced basis")
        st.write("New basis:")
        st.code(format_array(reduced_basis), language="text")

        st.write("Unimodular matrix U:")
        st.code(str(unimodular_matrix.tolist()), language="text")

        st.write("Relation:")
        st.code("B_reduced = B · U", language="text")

        st.write(f"det(U) = `{round(float(np.linalg.det(unimodular_matrix)))}`")


def main() -> None:
    st.set_page_config(
        page_title="LWE Lattice Visualizer",
        page_icon="🔷",
        layout="wide",
    )

    st.title("🔷 Lattice Visualizer")
    st.caption("Educational visualization of small lattices for SVP and CVP examples.")

    with st.sidebar:
        st.header("Input")

        basis_text = st.text_area(
            "Basis matrix B",
            value="[[2, 1],\n [0, 1]]",
            height=120,
            help="Basis vectors are columns of B. Example: [[2, 1], [0, 1]]",
        )

        radius = st.slider(
            "Coefficient range R for z ∈ [-R, R]ⁿ",
            min_value=1,
            max_value=10,
            value=5,
        )

        use_target = st.checkbox("Use target point for CVP", value=True)

        target_text = st.text_input(
            "Target point t",
            value="[2.3, 1.7]",
            help="Example for 2D: [2.3, 1.7]",
            disabled=not use_target,
        )

        show_reduced_basis = st.checkbox(
            "Show reduced basis for 2D",
            value=True,
        )

    try:
        B = parse_matrix(basis_text)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    n = B.shape[0]
    det_value = lattice_determinant(B)

    total_points = (2 * radius + 1) ** n

    if total_points > 200_000:
        st.error(
            f"Too many lattice points: {(2 * radius + 1)}^{n} = {total_points}. "
            "Please decrease the range or dimension."
        )
        st.stop()

    coefficient_vectors, lattice_points = generate_lattice_points(B, radius)

    shortest_coefficients, shortest_vector, shortest_length = find_shortest_vector(
        coefficient_vectors,
        lattice_points,
    )

    target = None
    closest_coefficients = None
    closest_point = None
    closest_distance = None

    if use_target:
        try:
            target = parse_vector(target_text, n)
            closest_coefficients, closest_point, closest_distance = find_closest_point(
                target,
                coefficient_vectors,
                lattice_points,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

    reduced_basis = None
    unimodular_matrix = None

    if n == 2 and show_reduced_basis:
        reduced_basis, unimodular_matrix = gauss_reduce_2d(B)

    left_col, right_col = st.columns([1, 2])

    with left_col:
        show_results(
            B=B,
            n=n,
            total_points=total_points,
            det_value=det_value,
            shortest_coefficients=shortest_coefficients,
            shortest_vector=shortest_vector,
            shortest_length=shortest_length,
            use_target=use_target,
            target=target,
            closest_coefficients=closest_coefficients,
            closest_point=closest_point,
            closest_distance=closest_distance,
            reduced_basis=reduced_basis,
            unimodular_matrix=unimodular_matrix,
        )

    with right_col:
        st.subheader("Visualization")

        if n == 2:
            fig = create_2d_plot(
                B=B,
                coefficient_vectors=coefficient_vectors,
                lattice_points=lattice_points,
                shortest_vector=shortest_vector,
                target=target,
                closest_point=closest_point,
                reduced_basis=reduced_basis,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif n > 2:
            st.warning(
                "Full geometric visualization is available only for 2D in this version. "
                "For higher dimensions, the app shows a 2D projection using the first two coordinates."
            )

            projected_points = lattice_points[:, :2]
            projected_B = B[:2, :2]

            projected_shortest = shortest_vector[:2]
            projected_target = target[:2] if target is not None else None
            projected_closest = closest_point[:2] if closest_point is not None else None

            fig = create_2d_plot(
                B=projected_B,
                coefficient_vectors=coefficient_vectors,
                lattice_points=projected_points,
                shortest_vector=projected_shortest,
                target=projected_target,
                closest_point=projected_closest,
                reduced_basis=None,
                title="2D Projection of the Lattice",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Visualization is available for dimension 2 or higher.")

    st.divider()

    st.write("### Notes")
    st.write(
        "The app uses brute force search inside the selected coefficient range. "
        "This is suitable for educational examples with small 2D/3D lattices, "
        "but it is not a professional SVP/CVP solver for high-dimensional cryptographic lattices."
    )


if __name__ == "__main__":
    main()