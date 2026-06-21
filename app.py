import numpy as np
import streamlit as st

from lattice.algorithms import (
    find_closest_point,
    find_shortest_vector,
    generate_basis_candidates,
)
from lattice.coordinates import coefficients_for_displayed_basis
from lattice.core import (
    estimate_coefficient_search_radius,
    generate_lattice_points_in_cube,
    lattice_determinant,
    parse_matrix,
    parse_vector,
)
from lattice.plotting_2d import create_2d_plot
from lattice.styles import apply_page_style
from lattice.ui_controls import (
    show_comparison_visualization_controls,
    show_normal_visualization_controls,
)
from lattice.ui_results import show_explanation_box, show_results
from lattice.ui_sidebar import show_sidebar
from lattice.visualization import (
    show_comparison_visualization,
    show_single_visualization,
)


def get_lattice_data_for_basis(
    B: np.ndarray,
    cube_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates lattice points for a selected basis.
    """
    search_radius = estimate_coefficient_search_radius(B, cube_limit)
    internal_candidates = (2 * search_radius + 1) ** B.shape[0]

    if internal_candidates > 200_000:
        st.error(
            f"Too many candidate coefficient vectors would be needed: "
            f"(2·{search_radius}+1)^{B.shape[0]} = {internal_candidates}. "
            f"Please decrease the cube limit or use a simpler basis."
        )
        st.stop()

    return generate_lattice_points_in_cube(
        B,
        cube_limit,
    )


def main() -> None:
    st.set_page_config(
        page_title="LWE Lattice Visualizer",
        page_icon="🔷",
        layout="wide",
    )
    apply_page_style()

    st.title("🔷 Lattice Visualizer")
    st.caption("Educational visualization of small lattices for SVP and CVP examples.")

    (
        _example_dimension,
        _selected_example,
        basis_text,
        cube_limit,
        use_target,
        target_text,
        highlight_mode,
    ) = show_sidebar()

    try:
        B_input = parse_matrix(basis_text)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    if B_input.shape in [(2, 2), (3, 3)]:
        convenient_bases, inconvenient_bases = generate_basis_candidates(
            B_input,
            limit=3,
            max_candidates=6,
        )
    else:
        convenient_bases = []
        inconvenient_bases = []

    current_comparison_mode = st.session_state.get("comparison_mode", False)

    if current_comparison_mode:
        (
            comparison_mode,
            B_left,
            left_label,
            left_unimodular_matrix,
            B_right,
            right_label,
            right_unimodular_matrix,
        ) = show_comparison_visualization_controls(
            B_input=B_input,
            convenient_bases=convenient_bases,
            inconvenient_bases=inconvenient_bases,
        )

        B_for_points = B_input
        B_for_results = B_right
        basis_mode_for_results = right_label
        unimodular_matrix_for_results = right_unimodular_matrix

    else:
        left_results_col, right_visualization_col = st.columns([1, 2])

        with right_visualization_col:
            (
                comparison_mode,
                B_displayed,
                basis_mode,
                unimodular_matrix,
            ) = show_normal_visualization_controls(
                B_input=B_input,
                convenient_bases=convenient_bases,
                inconvenient_bases=inconvenient_bases,
            )

        B_left = B_input
        left_label = "Input basis"
        left_unimodular_matrix = None
        B_right = B_displayed
        right_label = basis_mode
        right_unimodular_matrix = unimodular_matrix

        B_for_points = B_displayed
        B_for_results = B_displayed
        basis_mode_for_results = basis_mode
        unimodular_matrix_for_results = unimodular_matrix

    n = B_for_results.shape[0]
    det_value = lattice_determinant(B_for_results)

    coefficient_vectors, lattice_points = get_lattice_data_for_basis(
        B=B_for_points,
        cube_limit=cube_limit,
    )

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

    if comparison_mode:
        left_coefficient_vectors = coefficients_for_displayed_basis(
            coefficient_vectors,
            left_unimodular_matrix,
        )
        right_coefficient_vectors = coefficients_for_displayed_basis(
            coefficient_vectors,
            right_unimodular_matrix,
        )
        results_shortest_coefficients = coefficients_for_displayed_basis(
            shortest_coefficients.reshape(1, -1),
            unimodular_matrix_for_results,
        )[0]

        if closest_coefficients is not None:
            results_closest_coefficients = coefficients_for_displayed_basis(
                closest_coefficients.reshape(1, -1),
                unimodular_matrix_for_results,
            )[0]
        else:
            results_closest_coefficients = None
    else:
        left_coefficient_vectors = coefficient_vectors
        right_coefficient_vectors = coefficient_vectors
        results_shortest_coefficients = shortest_coefficients
        results_closest_coefficients = closest_coefficients

    if n in [2, 3]:
        if comparison_mode:
            show_comparison_visualization(
                n=n,
                B_left=B_left,
                left_label=left_label,
                left_coefficient_vectors=left_coefficient_vectors,
                B_right=B_right,
                right_label=right_label,
                right_coefficient_vectors=right_coefficient_vectors,
                lattice_points=lattice_points,
                shortest_vector=shortest_vector,
                target=target,
                closest_point=closest_point,
                highlight_mode=highlight_mode,
            )

            with st.expander("Show numerical results", expanded=False):
                show_results(
                    B_displayed=B_for_results,
                    det_value=det_value,
                    shortest_coefficients=results_shortest_coefficients,
                    shortest_vector=shortest_vector,
                    shortest_length=shortest_length,
                    use_target=use_target,
                    target=target,
                    closest_coefficients=results_closest_coefficients,
                    closest_point=closest_point,
                    closest_distance=closest_distance,
                    basis_mode=basis_mode_for_results,
                    unimodular_matrix=unimodular_matrix_for_results,
                )

        else:
            with left_results_col:
                show_results(
                    B_displayed=B_for_results,
                    det_value=det_value,
                    shortest_coefficients=results_shortest_coefficients,
                    shortest_vector=shortest_vector,
                    shortest_length=shortest_length,
                    use_target=use_target,
                    target=target,
                    closest_coefficients=results_closest_coefficients,
                    closest_point=closest_point,
                    closest_distance=closest_distance,
                    basis_mode=basis_mode_for_results,
                    unimodular_matrix=unimodular_matrix_for_results,
                )

            with right_visualization_col:
                show_single_visualization(
                    n=n,
                    B_displayed=B_for_results,
                    basis_label=basis_mode_for_results,
                    coefficient_vectors=coefficient_vectors,
                    lattice_points=lattice_points,
                    shortest_vector=shortest_vector,
                    target=target,
                    closest_point=closest_point,
                    highlight_mode=highlight_mode,
                )

    elif n > 3:
        if comparison_mode:
            st.warning(
                "Comparison mode is available only for full 2D and 3D visualization. "
                "For higher dimensions, the app shows only a 2D projection."
            )
            left_results_col, right_visualization_col = st.columns([1, 2])

        with left_results_col:
            show_results(
                B_displayed=B_for_results,
                det_value=det_value,
                shortest_coefficients=results_shortest_coefficients,
                shortest_vector=shortest_vector,
                shortest_length=shortest_length,
                use_target=use_target,
                target=target,
                closest_coefficients=results_closest_coefficients,
                closest_point=closest_point,
                closest_distance=closest_distance,
                basis_mode=basis_mode_for_results,
                unimodular_matrix=unimodular_matrix_for_results,
            )

        with right_visualization_col:
            st.warning(
                "Full geometric visualization is available only for 2D and 3D. "
                "For higher dimensions, the app shows a 2D projection using the first two coordinates."
            )

            projected_points = lattice_points[:, :2]
            projected_B = B_for_results[:2, :2]

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
                highlight_mode=highlight_mode,
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Visualization is available for dimension 2 or higher.")

    st.divider()
    show_explanation_box(n)


if __name__ == "__main__":
    main()