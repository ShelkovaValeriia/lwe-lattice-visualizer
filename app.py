from typing import Optional

import numpy as np
import streamlit as st

from lattice.algorithms import (
    find_closest_point,
    find_shortest_vector,
    generate_basis_candidates_2d,
)
from lattice.core import (
    estimate_coefficient_search_radius,
    generate_lattice_points_in_cube,
    lattice_determinant,
    parse_matrix,
    parse_vector,
)
from lattice.plotting_2d import create_2d_plot
from lattice.plotting_3d import create_3d_plot
from lattice.utils import format_array


def show_basis_vectors(B: np.ndarray) -> None:
    """
    Shows basis vectors as columns of the basis matrix.
    """
    st.write("### Displayed basis vectors")

    for i in range(B.shape[1]):
        st.write(f"b{i + 1} = `{format_array(B[:, i])}`")


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
    Shows compact numerical results in the left column.
    """
    st.subheader("Results")

    st.metric("|det(B)|", round(det_value, 6))

    st.write("### Basis mode")
    st.write(basis_mode)

    show_basis_vectors(B_displayed)

    if unimodular_matrix is not None:
        st.write("### Same lattice relation")
        st.code("B_new = B · U", language="text")

        st.write("Unimodular matrix U:")
        st.code(str(unimodular_matrix.tolist()), language="text")

        st.write(f"det(U) = `{round(float(np.linalg.det(unimodular_matrix)))}`")

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


def get_default_basis_and_target(example_dimension: str) -> tuple[str, str]:
    """
    Returns default basis and target point for selected example.
    """
    if example_dimension == "2D":
        return (
            "[[2, 1],\n [0, 1]]",
            "[2.3, 1.7]",
        )

    return (
        "[[1, 0, 1],\n [0, 1, 1],\n [0, 0, 2]]",
        "[1.4, 1.6, 2.2]",
    )


def choose_displayed_basis(B: np.ndarray) -> tuple[np.ndarray, str, Optional[np.ndarray]]:
    """
    Lets the user choose which basis should be displayed.

    All generated 2D bases define the same lattice as the input basis.
    """
    if B.shape != (2, 2):
        return B, "Input basis", None

    st.sidebar.write("---")
    st.sidebar.subheader("Same lattice bases")

    convenient_bases, inconvenient_bases = generate_basis_candidates_2d(
        B,
        limit=3,
        max_candidates=6,
    )

    basis_mode = st.sidebar.selectbox(
        "Displayed basis",
        options=[
            "Input basis",
            "Convenient basis",
            "Inconvenient basis",
        ],
        index=0,
        help=(
            "Different bases are generated as B_new = B · U, "
            "where U is unimodular and det(U) = ±1. "
            "Therefore the lattice stays the same."
        ),
    )

    if basis_mode == "Input basis":
        return B, "Input basis", None

    if basis_mode == "Convenient basis":
        options = convenient_bases
        label = "Convenient basis"

    else:
        options = inconvenient_bases
        label = "Inconvenient basis"

    option_labels = []

    for index, item in enumerate(options):
        basis = item["basis"]
        score = item["score"]

        b1_len = np.linalg.norm(basis[:, 0])
        b2_len = np.linalg.norm(basis[:, 1])

        option_labels.append(
            f"{index + 1}. score={score:.3f}, |b1|={b1_len:.3f}, |b2|={b2_len:.3f}"
        )

    selected_index = st.sidebar.selectbox(
        "Candidate",
        options=list(range(len(option_labels))),
        format_func=lambda i: option_labels[i],
        index=0,
    )

    selected = options[selected_index]

    return (
        selected["basis"],
        label,
        selected["U"],
    )


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

        example_dimension = st.selectbox(
            "Example dimension",
            options=["2D", "3D"],
            index=0,
            help="This only changes the default example. You can still enter your own matrix.",
        )

        default_basis, default_target = get_default_basis_and_target(example_dimension)

        basis_text = st.text_area(
            "Basis matrix B",
            value=default_basis,
            height=140,
            help="Basis vectors are columns of B. Example: [[2, 1], [0, 1]]",
        )

        cube_limit = st.slider(
            "Coordinate cube limit L for each axis",
            min_value=1,
            max_value=10,
            value=2,
            help="The app shows lattice points whose coordinates lie in [-L, L] on every axis.",
        )

        use_target = st.checkbox("Use target point for CVP", value=True)

        target_text = st.text_input(
            "Target point t",
            value=default_target,
            help="Example for 2D: [2.3, 1.7], example for 3D: [1.4, 1.6, 2.2]",
            disabled=not use_target,
        )

    try:
        B_input = parse_matrix(basis_text)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    B_displayed, basis_mode, unimodular_matrix = choose_displayed_basis(B_input)

    n = B_displayed.shape[0]
    det_value = lattice_determinant(B_displayed)

    search_radius = estimate_coefficient_search_radius(B_displayed, cube_limit)
    internal_candidates = (2 * search_radius + 1) ** n

    if internal_candidates > 200_000:
        st.error(
            f"Too many candidate coefficient vectors would be needed: "
            f"(2·{search_radius}+1)^{n} = {internal_candidates}. "
            f"Please decrease the cube limit or use a simpler basis."
        )
        st.stop()

    coefficient_vectors, lattice_points = generate_lattice_points_in_cube(
        B_displayed,
        cube_limit,
    )

    total_points = len(lattice_points)

    non_zero_mask = np.any(coefficient_vectors != 0, axis=1)

    if not np.any(non_zero_mask):
        st.error(
            "Only the origin lies inside the selected cube. "
            "Please increase the cube limit."
        )
        st.stop()

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

    left_col, right_col = st.columns([1, 2])

    with left_col:
        show_results(
            B_displayed=B_displayed,
            det_value=det_value,
            shortest_coefficients=shortest_coefficients,
            shortest_vector=shortest_vector,
            shortest_length=shortest_length,
            use_target=use_target,
            target=target,
            closest_coefficients=closest_coefficients,
            closest_point=closest_point,
            closest_distance=closest_distance,
            basis_mode=basis_mode,
            unimodular_matrix=unimodular_matrix,
        )

    with right_col:
        st.subheader("Visualization")

        if n == 2:
            fig = create_2d_plot(
                B=B_displayed,
                coefficient_vectors=coefficient_vectors,
                lattice_points=lattice_points,
                shortest_vector=shortest_vector,
                target=target,
                closest_point=closest_point,
                reduced_basis=None,
            )

            st.plotly_chart(fig, use_container_width=True)

        elif n == 3:
            fig = create_3d_plot(
                B=B_displayed,
                coefficient_vectors=coefficient_vectors,
                lattice_points=lattice_points,
                shortest_vector=shortest_vector,
                target=target,
                closest_point=closest_point,
            )

            st.plotly_chart(fig, use_container_width=True)

        elif n > 3:
            st.warning(
                "Full geometric visualization is available only for 2D and 3D. "
                "For higher dimensions, the app shows a 2D projection using the first two coordinates."
            )

            projected_points = lattice_points[:, :2]
            projected_B = B_displayed[:2, :2]

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
        "The visualization limit defines a coordinate cube [-L, L]^n in the ambient space. "
        "The app internally searches for coefficient vectors z, constructs lattice points Bz, "
        "and keeps only those points whose coordinates lie inside the selected cube."
    )

    st.write(
        "For 2D lattices, the app can display different bases of the same lattice. "
        "A new basis is constructed as B_new = B · U, where U is a unimodular integer matrix "
        "with det(U) = ±1."
    )


if __name__ == "__main__":
    main()