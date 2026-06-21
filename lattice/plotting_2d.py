from typing import Optional

import numpy as np
import plotly.graph_objects as go


PLOT_BG = "#0E1117"
PAPER_BG = "#0E1117"
GRID_COLOR = "rgba(180, 200, 255, 0.18)"
AXIS_COLOR = "rgba(255, 255, 255, 0.75)"
POINT_COLOR = "rgba(235, 235, 235, 0.85)"


HIGHLIGHT_MODES = {"All", "SVP", "CVP", "Basis", "None"}


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
            textfont=dict(color=color, size=14),
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
    highlight_mode: str = "All",
) -> go.Figure:
    """
    Creates a 2D Plotly visualization of the lattice.
    """
    if highlight_mode not in HIGHLIGHT_MODES:
        highlight_mode = "All"

    show_basis = highlight_mode in ["All", "Basis"]
    show_svp = highlight_mode in ["All", "SVP"]
    show_cvp = highlight_mode in ["All", "CVP"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=lattice_points[:, 0],
            y=lattice_points[:, 1],
            mode="markers",
            name="Lattice points",
            marker=dict(
                size=7,
                color=POINT_COLOR,
                line=dict(width=1, color="rgba(255,255,255,0.35)"),
            ),
            text=[
                f"z = {z.tolist()}<br>Bz = {np.round(p, 4).tolist()}"
                for z, p in zip(coefficient_vectors, lattice_points)
            ],
            hoverinfo="text",
        )
    )

    b1 = B[:, 0]
    b2 = B[:, 1]

    if show_basis:
        parallelogram_x = [0, b1[0], b1[0] + b2[0], b2[0], 0]
        parallelogram_y = [0, b1[1], b1[1] + b2[1], b2[1], 0]

        fig.add_trace(
            go.Scatter(
                x=parallelogram_x,
                y=parallelogram_y,
                mode="lines",
                fill="toself",
                name="Fundamental parallelogram",
                line=dict(color="rgba(0, 150, 255, 0.95)", width=3),
                fillcolor="rgba(0, 150, 255, 0.16)",
            )
        )

        add_2d_vector(fig, b1, "Basis vector b₁", "#1E90FF", label="b₁")
        add_2d_vector(fig, b2, "Basis vector b₂", "#00C853", label="b₂")

    if show_basis and reduced_basis is not None:
        r1 = reduced_basis[:, 0]
        r2 = reduced_basis[:, 1]

        add_2d_vector(
            fig,
            r1,
            "Reduced basis vector r₁",
            "#B388FF",
            width=3,
            dash="dash",
            label="r₁",
        )

        add_2d_vector(
            fig,
            r2,
            "Reduced basis vector r₂",
            "#B388FF",
            width=3,
            dash="dash",
            label="r₂",
        )

    if show_svp and shortest_vector is not None:
        add_2d_vector(
            fig,
            shortest_vector,
            "Shortest vector",
            "#FF1744",
            width=5,
            label="SVP",
        )

    if show_cvp and target is not None:
        fig.add_trace(
            go.Scatter(
                x=[target[0]],
                y=[target[1]],
                mode="markers+text",
                name="Target point t",
                marker=dict(size=14, color="#FFC400", symbol="x"),
                text=["t"],
                textposition="top center",
                textfont=dict(color="#FFC400", size=14),
            )
        )

    if show_cvp and target is not None and closest_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[closest_point[0]],
                y=[closest_point[1]],
                mode="markers+text",
                name="Closest lattice point",
                marker=dict(size=13, color="#FF1744", symbol="diamond"),
                text=["closest"],
                textposition="bottom center",
                textfont=dict(color="#FF1744", size=13),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[target[0], closest_point[0]],
                y=[target[1], closest_point[1]],
                mode="lines",
                name="Distance to closest point",
                line=dict(color="#FFC400", width=3, dash="dot"),
            )
        )

    fig.add_hline(y=0, line_width=2, line_color=AXIS_COLOR, opacity=0.8)
    fig.add_vline(x=0, line_width=2, line_color=AXIS_COLOR, opacity=0.8)

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=18)),
        xaxis_title="x",
        yaxis_title="y",
        width=900,
        height=700,
        showlegend=True,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(14, 17, 23, 0.75)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            gridcolor=GRID_COLOR,
            zerolinecolor=AXIS_COLOR,
            linecolor=AXIS_COLOR,
            tickfont=dict(color="white"),
            title=dict(font=dict(color="white")),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=AXIS_COLOR,
            linecolor=AXIS_COLOR,
            tickfont=dict(color="white"),
            title=dict(font=dict(color="white")),
        ),
    )

    return fig