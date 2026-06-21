from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lattice.plotting_2d import create_2d_plot
from lattice.plotting_3d import create_3d_plot


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
    left_coefficient_vectors: np.ndarray,
    B_right: np.ndarray,
    right_label: str,
    right_coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
    shortest_vector: np.ndarray,
    target: Optional[np.ndarray],
    closest_point: Optional[np.ndarray],
    highlight_mode: str,
) -> None:
    """
    Shows two independently selected bases side by side.

    Both graphs receive exactly the same lattice points.
    Only basis vectors, fundamental domains, and coordinate labels are different.
    """
    st.info(
        "Comparison mode is active. Both plots show the same lattice points and use the same scale. "
        "Only the displayed basis and coefficient coordinates change."
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
            coefficient_vectors=left_coefficient_vectors,
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
            coefficient_vectors=right_coefficient_vectors,
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