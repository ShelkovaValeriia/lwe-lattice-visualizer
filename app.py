from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lattice.algorithms import (
    find_closest_point,
    find_shortest_vector,
    generate_basis_candidates,
)
from lattice.core import (
    estimate_coefficient_search_radius,
    generate_lattice_points_in_cube,
    lattice_determinant,
    parse_matrix,
    parse_vector,
)
from lattice.examples import EXAMPLES_BY_DIMENSION
from lattice.plotting_2d import create_2d_plot
from lattice.plotting_3d import create_3d_plot
from lattice.utils import format_array


BASIS_MODES = [
    "Input basis",
    "Convenient basis",
    "Inconvenient basis",
]


def apply_page_style() -> None:
    """
    Makes the Streamlit layout more compact and removes unnecessary empty space.
    """
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2rem;
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.2rem;
                padding-bottom: 1.5rem;
            }

            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                margin-top: 0.35rem;
                margin-bottom: 0.55rem;
            }

            section[data-testid="stSidebar"] hr {
                margin-top: 1rem;
                margin-bottom: 1rem;
            }

            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 0.75rem;
                padding: 0.75rem 0.9rem;
            }

            div[data-testid="stAlert"] {
                border-radius: 0.7rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def create_lattice_plot(
    n: int,
    B: np.ndarray,
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
    shortest_vector: Optional[np.ndarray],
    target: Optional[np.ndarray],
    closest_point: Optional[np.ndarray],
    highlight_mode: str,
    title: str,
) -> go.Figure:
    """
    Creates either a 2D or a 3D plot depending on the lattice dimension.
    """
    if n == 2:
        return create_2d_plot(
            B=B,
            coefficient_vectors=coefficient_vectors,
            lattice_points=lattice_points,
            shortest_vector=shortest_vector,
            target=target,
            closest_point=closest_point,
            reduced_basis=None,
            title=title,
            highlight_mode=highlight_mode,
        )

    return create_3d_plot(
        B=B,
        coefficient_vectors=coefficient_vectors,
        lattice_points=lattice_points,
        shortest_vector=shortest_vector,
        target=target,
        closest_point=closest_point,
        title=title,
        highlight_mode=highlight_mode,
    )


def collect_points_for_range(
    B_left: np.ndarray,
    B_right: np.ndarray,
    lattice_points: np.ndarray,
    target: Optional[np.ndarray],
    closest_point: Optional[np.ndarray],
    shortest_vector: Optional[np.ndarray],
) -> np.ndarray:
    """
    Collects all important points so both comparison plots can use the same range.
    """
    points = [lattice_points]

    important_points = [
        np.zeros(B_left.shape[0]),
        *[B_left[:, i] for i in range(B_left.shape[1])],
        *[B_right[:, i] for i in range(B_right.shape[1])],
    ]

    if target is not None:
        important_points.append(target)

    if closest_point is not None:
        important_points.append(closest_point)

    if shortest_vector is not None:
        important_points.append(shortest_vector)

    points.append(np.array(important_points))

    return np.vstack(points)


def get_common_axis_range(points: np.ndarray) -> list[float]:
    """
    Creates one common symmetric range for x/y or x/y/z axes.
    """
    max_abs = max(1.0, float(np.max(np.abs(points))))
    margin = max_abs * 0.15
    axis_limit = max_abs + margin

    return [-axis_limit, axis_limit]


def apply_common_range_to_2d_plot(
    fig: go.Figure,
    axis_range: list[float],
    height: int,
) -> go.Figure:
    """
    Applies the same x/y range to a 2D plot.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    fig.update_xaxes(range=axis_range)
    fig.update_yaxes(range=axis_range)

    return fig


def apply_common_range_to_3d_plot(
    fig: go.Figure,
    axis_range: list[float],
    height: int,
) -> go.Figure:
    """
    Applies the same x/y/z range to a 3D plot.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        scene=dict(
            xaxis=dict(range=axis_range),
            yaxis=dict(range=axis_range),
            zaxis=dict(range=axis_range),
        ),
    )

    return fig


def prepare_plot_for_layout(
    fig: go.Figure,
    n: int,
    axis_range: Optional[list[float]],
    height: int,
) -> go.Figure:
    """
    Adjusts plot size and common range before displaying.
    """
    if axis_range is None:
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=10, t=55, b=10),
        )

        return fig

    if n == 2:
        return apply_common_range_to_2d_plot(fig, axis_range, height)

    if n == 3:
        return apply_common_range_to_3d_plot(fig, axis_range, height)

    return fig


def show_single_visualization(
    n: int,
    B_displayed: np.ndarray,
    basis_label: str,
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
    shortest_vector: np.ndarray,
    target: Optional[np.ndarray],
    closest_point: Optional[np.ndarray],
    highlight_mode: str,
) -> None:
    """
    Shows one visualization of the currently selected basis.
    """
    fig = create_lattice_plot(
        n=n,
        B=B_displayed,
        coefficient_vectors=coefficient_vectors,
        lattice_points=lattice_points,
        shortest_vector=shortest_vector,
        target=target,
        closest_point=closest_point,
        highlight_mode=highlight_mode,
        title=basis_label,
    )

    fig = prepare_plot_for_layout(
        fig=fig,
        n=n,
        axis_range=None,
        height=720,
    )

    st.plotly_chart(fig, use_container_width=True)


def show_comparison_visualization(
    n: int,
    B_left: np.ndarray,
    left_label: str,
    B_right: np.ndarray,
    right_label: str,
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
    shortest_vector: np.ndarray,
    target: Optional[np.ndarray],
    closest_point: Optional[np.ndarray],
    highlight_mode: str,
) -> None:
    """
    Shows two independently selected bases side by side.

    Both graphs receive exactly the same lattice points.
    Only basis vectors and the fundamental domain are different.
    """
    st.info(
        "Comparison mode is active. Both plots use the same lattice points and the same scale. "
        "Only basis vectors and the fundamental domain change."
    )

    all_range_points = collect_points_for_range(
        B_left=B_left,
        B_right=B_right,
        lattice_points=lattice_points,
        target=target,
        closest_point=closest_point,
        shortest_vector=shortest_vector,
    )

    axis_range = get_common_axis_range(all_range_points)

    left_col, right_col = st.columns(2)

    with left_col:
        left_fig = create_lattice_plot(
            n=n,
            B=B_left,
            coefficient_vectors=coefficient_vectors,
            lattice_points=lattice_points,
            shortest_vector=shortest_vector,
            target=target,
            closest_point=closest_point,
            highlight_mode=highlight_mode,
            title=left_label,
        )

        left_fig = prepare_plot_for_layout(
            fig=left_fig,
            n=n,
            axis_range=axis_range,
            height=670,
        )

        st.plotly_chart(left_fig, use_container_width=True)

    with right_col:
        right_fig = create_lattice_plot(
            n=n,
            B=B_right,
            coefficient_vectors=coefficient_vectors,
            lattice_points=lattice_points,
            shortest_vector=shortest_vector,
            target=target,
            closest_point=closest_point,
            highlight_mode=highlight_mode,
            title=right_label,
        )

        right_fig = prepare_plot_for_layout(
            fig=right_fig,
            n=n,
            axis_range=axis_range,
            height=670,
        )

        st.plotly_chart(right_fig, use_container_width=True)


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


def show_sidebar() -> tuple[str, dict, str, int, bool, str, str]:
    """
    Shows all input controls in the sidebar.
    """
    with st.sidebar:
        st.header("Input")

        example_dimension = st.selectbox(
            "Dimension",
            options=["2D", "3D"],
            index=0,
            help="Choose whether to work with a 2D or 3D lattice.",
        )

        examples_for_dimension = EXAMPLES_BY_DIMENSION[example_dimension]

        selected_example_name = st.selectbox(
            "Preset example",
            options=list(examples_for_dimension.keys()),
            index=1 if example_dimension == "2D" else 0,
            help="Choose a prepared example. You can still edit the matrix manually.",
        )

        selected_example = examples_for_dimension[selected_example_name]

        st.divider()
        st.subheader("Basis and range")

        basis_text = st.text_area(
            "Basis matrix B",
            value=selected_example["basis"],
            height=130,
            help="Basis vectors are columns of B. Example: [[2, 1], [0, 1]]",
        )

        cube_limit = st.slider(
            "Visible coordinate limit L",
            min_value=1,
            max_value=10,
            value=selected_example["cube_limit"],
            help="The app shows lattice points whose coordinates lie in [-L, L] on every axis.",
        )

        st.divider()
        st.subheader("CVP target")

        use_target = st.checkbox("Show target point", value=True)

        target_text = st.text_input(
            "Target point t",
            value=selected_example["target"],
            help="Example for 2D: [2.3, 1.7], example for 3D: [1.4, 1.6, 2.2]",
            disabled=not use_target,
        )

        st.divider()
        st.subheader("Plot")

        highlight_mode = st.selectbox(
            "Highlight",
            options=["All", "SVP", "CVP", "Basis", "None"],
            index=0,
            help=(
                "Choose which additional elements should be highlighted on the plot. "
                "Lattice points are always visible."
            ),
        )

    return (
        example_dimension,
        selected_example,
        basis_text,
        cube_limit,
        use_target,
        target_text,
        highlight_mode,
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

        B_for_results = B_right
        basis_mode_for_results = right_label
        unimodular_matrix_for_results = right_unimodular_matrix
        B_for_points = B_input

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
        B_right = B_displayed
        right_label = basis_mode

        B_for_results = B_displayed
        basis_mode_for_results = basis_mode
        unimodular_matrix_for_results = unimodular_matrix
        B_for_points = B_displayed

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

    if n in [2, 3]:
        if comparison_mode:
            show_comparison_visualization(
                n=n,
                B_left=B_left,
                left_label=left_label,
                B_right=B_right,
                right_label=right_label,
                coefficient_vectors=coefficient_vectors,
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
                    shortest_coefficients=shortest_coefficients,
                    shortest_vector=shortest_vector,
                    shortest_length=shortest_length,
                    use_target=use_target,
                    target=target,
                    closest_coefficients=closest_coefficients,
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
                    shortest_coefficients=shortest_coefficients,
                    shortest_vector=shortest_vector,
                    shortest_length=shortest_length,
                    use_target=use_target,
                    target=target,
                    closest_coefficients=closest_coefficients,
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
                shortest_coefficients=shortest_coefficients,
                shortest_vector=shortest_vector,
                shortest_length=shortest_length,
                use_target=use_target,
                target=target,
                closest_coefficients=closest_coefficients,
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