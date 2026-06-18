from typing import Optional

import numpy as np
import plotly.graph_objects as go


def add_2d_vector(
    fig: go.Figure,
    vector: np.ndarray,
    name: str,
    color: str,
    width: int = 4,
    dash: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    """
    Adds a 2D vector as a line from the origin.
    """
    line_settings = dict(color=color, width=width)

    if dash is not None:
        line_settings["dash"] = dash

    fig.add_trace(
        go.Scatter(
            x=[0, vector[0]],
            y=[0, vector[1]],
            mode="lines+markers+text",
            name=name,
            line=line_settings,
            marker=dict(size=8, color=color),
            text=["", label if label is not None else name],
            textposition="top center",
        )
    )


def create_2d_plot(
    B: np.ndarray,
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
    shortest_vector: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    closest_point: Optional[np.ndarray] = None,
    reduced_basis: Optional[np.ndarray] = None,
    title: str = "2D Lattice Visualization",
) -> go.Figure:
    """
    Creates a 2D Plotly visualization of the lattice.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=lattice_points[:, 0],
            y=lattice_points[:, 1],
            mode="markers",
            name="Lattice points",
            marker=dict(size=7, color="lightgray", line=dict(width=1, color="gray")),
            text=[
                f"z = {z.tolist()}<br>Bz = {np.round(p, 4).tolist()}"
                for z, p in zip(coefficient_vectors, lattice_points)
            ],
            hoverinfo="text",
        )
    )

    b1 = B[:, 0]
    b2 = B[:, 1]

    parallelogram_x = [0, b1[0], b1[0] + b2[0], b2[0], 0]
    parallelogram_y = [0, b1[1], b1[1] + b2[1], b2[1], 0]

    fig.add_trace(
        go.Scatter(
            x=parallelogram_x,
            y=parallelogram_y,
            mode="lines",
            fill="toself",
            name="Fundamental parallelogram",
            line=dict(color="rgba(0, 100, 255, 0.7)", width=2),
            fillcolor="rgba(0, 100, 255, 0.15)",
        )
    )

    add_2d_vector(fig, b1, "Basis vector b₁", "blue", label="b₁")
    add_2d_vector(fig, b2, "Basis vector b₂", "green", label="b₂")

    if reduced_basis is not None:
        r1 = reduced_basis[:, 0]
        r2 = reduced_basis[:, 1]

        add_2d_vector(
            fig,
            r1,
            "Reduced basis vector r₁",
            "purple",
            width=3,
            dash="dash",
            label="r₁",
        )

        add_2d_vector(
            fig,
            r2,
            "Reduced basis vector r₂",
            "purple",
            width=3,
            dash="dash",
            label="r₂",
        )

    if shortest_vector is not None:
        add_2d_vector(
            fig,
            shortest_vector,
            "Shortest vector",
            "red",
            width=5,
            label="SVP",
        )

    if target is not None:
        fig.add_trace(
            go.Scatter(
                x=[target[0]],
                y=[target[1]],
                mode="markers+text",
                name="Target point t",
                marker=dict(size=13, color="orange", symbol="x"),
                text=["t"],
                textposition="top center",
            )
        )

    if target is not None and closest_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[closest_point[0]],
                y=[closest_point[1]],
                mode="markers+text",
                name="Closest lattice point",
                marker=dict(size=13, color="red", symbol="diamond"),
                text=["closest"],
                textposition="bottom center",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[target[0], closest_point[0]],
                y=[target[1], closest_point[1]],
                mode="lines",
                name="Distance to closest point",
                line=dict(color="orange", width=3, dash="dot"),
            )
        )

    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.4)
    fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.4)

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        width=900,
        height=700,
        showlegend=True,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
    )

    return fig