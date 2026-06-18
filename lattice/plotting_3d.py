from typing import Optional

import numpy as np
import plotly.graph_objects as go


PLOT_BG = "#0E1117"
GRID_COLOR = "rgba(180, 200, 255, 0.18)"
AXIS_COLOR = "rgba(255, 255, 255, 0.75)"
POINT_COLOR = "rgba(235, 235, 235, 0.70)"


def add_3d_vector(
    fig: go.Figure,
    vector: np.ndarray,
    name: str,
    color: str,
    width: int = 6,
    label: Optional[str] = None,
) -> None:
    """
    Adds a 3D vector as a line from the origin.
    """
    fig.add_trace(
        go.Scatter3d(
            x=[0, vector[0]],
            y=[0, vector[1]],
            z=[0, vector[2]],
            mode="lines+markers+text",
            name=name,
            line=dict(color=color, width=width),
            marker=dict(size=4, color=color),
            text=["", label if label is not None else name],
            textposition="top center",
            textfont=dict(color=color, size=13),
        )
    )


def add_3d_line(
    fig: go.Figure,
    start: np.ndarray,
    end: np.ndarray,
    name: str,
    color: str,
    width: int = 4,
    dash: Optional[str] = None,
    showlegend: bool = False,
) -> None:
    """
    Adds a 3D line segment.
    """
    line_settings = dict(color=color, width=width)

    if dash is not None:
        line_settings["dash"] = dash

    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            name=name,
            line=line_settings,
            showlegend=showlegend,
        )
    )


def add_coordinate_axes(fig: go.Figure, axis_limit: float) -> None:
    """
    Adds 3D coordinate axes through the origin.
    """
    axis_style = "rgba(255, 255, 255, 0.85)"

    # X axis
    fig.add_trace(
        go.Scatter3d(
            x=[-axis_limit, axis_limit],
            y=[0, 0],
            z=[0, 0],
            mode="lines+text",
            name="x-axis",
            line=dict(color=axis_style, width=5),
            text=["", "x"],
            textposition="top center",
            textfont=dict(color="white", size=14),
            showlegend=False,
        )
    )

    # Y axis
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[-axis_limit, axis_limit],
            z=[0, 0],
            mode="lines+text",
            name="y-axis",
            line=dict(color=axis_style, width=5),
            text=["", "y"],
            textposition="top center",
            textfont=dict(color="white", size=14),
            showlegend=False,
        )
    )

    # Z axis
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-axis_limit, axis_limit],
            mode="lines+text",
            name="z-axis",
            line=dict(color=axis_style, width=5),
            text=["", "z"],
            textposition="top center",
            textfont=dict(color="white", size=14),
            showlegend=False,
        )
    )

    # Origin
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers+text",
            name="origin",
            marker=dict(size=4, color="white"),
            text=["O"],
            textposition="bottom center",
            textfont=dict(color="white", size=12),
            showlegend=False,
        )
    )


def add_fundamental_parallelepiped(
    fig: go.Figure,
    B: np.ndarray,
) -> None:
    """
    Adds the fundamental parallelepiped for a 3D lattice.

    The vertices are:
    0, b1, b2, b3, b1+b2, b1+b3, b2+b3, b1+b2+b3.
    """
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    vertices = [
        np.array([0.0, 0.0, 0.0]),
        b1,
        b2,
        b3,
        b1 + b2,
        b1 + b3,
        b2 + b3,
        b1 + b2 + b3,
    ]

    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]

    for index, (i, j) in enumerate(edges):
        add_3d_line(
            fig=fig,
            start=vertices[i],
            end=vertices[j],
            name="Fundamental parallelepiped",
            color="rgba(0, 150, 255, 0.95)",
            width=5,
            showlegend=index == 0,
        )

    mesh_vertices = np.array(vertices)

    # Six faces of the parallelepiped, each split into two triangles.
    i_faces = [0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 3, 3]
    j_faces = [1, 4, 1, 5, 4, 7, 4, 7, 2, 6, 5, 7]
    k_faces = [4, 2, 5, 3, 7, 5, 7, 6, 6, 3, 7, 6]

    fig.add_trace(
        go.Mesh3d(
            x=mesh_vertices[:, 0],
            y=mesh_vertices[:, 1],
            z=mesh_vertices[:, 2],
            i=i_faces,
            j=j_faces,
            k=k_faces,
            name="Fundamental volume",
            opacity=0.18,
            color="#1E90FF",
            showlegend=True,
        )
    )


def create_3d_plot(
    B: np.ndarray,
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
    shortest_vector: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    closest_point: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Creates a 3D Plotly visualization of the lattice.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=lattice_points[:, 0],
            y=lattice_points[:, 1],
            z=lattice_points[:, 2],
            mode="markers",
            name="Lattice points",
            marker=dict(
                size=4,
                color=POINT_COLOR,
                opacity=0.8,
                line=dict(width=0.5, color="rgba(255,255,255,0.25)"),
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
    b3 = B[:, 2]

    add_3d_vector(fig, b1, "Basis vector b₁", "#1E90FF", label="b₁")
    add_3d_vector(fig, b2, "Basis vector b₂", "#00C853", label="b₂")
    add_3d_vector(fig, b3, "Basis vector b₃", "#B388FF", label="b₃")

    add_fundamental_parallelepiped(fig, B)

    if shortest_vector is not None:
        add_3d_vector(
            fig,
            shortest_vector,
            "Shortest vector",
            "#FF1744",
            width=8,
            label="SVP",
        )

    if target is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[target[0]],
                y=[target[1]],
                z=[target[2]],
                mode="markers+text",
                name="Target point t",
                marker=dict(size=7, color="#FFC400", symbol="x"),
                text=["t"],
                textposition="top center",
                textfont=dict(color="#FFC400", size=13),
            )
        )

    if target is not None and closest_point is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[closest_point[0]],
                y=[closest_point[1]],
                z=[closest_point[2]],
                mode="markers+text",
                name="Closest lattice point",
                marker=dict(size=7, color="#FF1744", symbol="diamond"),
                text=["closest"],
                textposition="bottom center",
                textfont=dict(color="#FF1744", size=13),
            )
        )

        add_3d_line(
            fig=fig,
            start=target,
            end=closest_point,
            name="Distance to closest point",
            color="#FFC400",
            width=6,
            dash="dot",
            showlegend=True,
        )

    max_abs = max(1.0, float(np.max(np.abs(lattice_points))))
    add_coordinate_axes(fig, max_abs)

    fig.update_layout(
        title=dict(text="3D Lattice Visualization", font=dict(color="white", size=18)),
        width=900,
        height=750,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="white"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(14, 17, 23, 0.75)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        scene=dict(
            bgcolor=PLOT_BG,
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
            xaxis=dict(
                range=[-max_abs, max_abs],
                backgroundcolor=PLOT_BG,
                gridcolor=GRID_COLOR,
                zerolinecolor=AXIS_COLOR,
                linecolor=AXIS_COLOR,
                tickfont=dict(color="white"),
                title=dict(font=dict(color="white")),
            ),
            yaxis=dict(
                range=[-max_abs, max_abs],
                backgroundcolor=PLOT_BG,
                gridcolor=GRID_COLOR,
                zerolinecolor=AXIS_COLOR,
                linecolor=AXIS_COLOR,
                tickfont=dict(color="white"),
                title=dict(font=dict(color="white")),
            ),
            zaxis=dict(
                range=[-max_abs, max_abs],
                backgroundcolor=PLOT_BG,
                gridcolor=GRID_COLOR,
                zerolinecolor=AXIS_COLOR,
                linecolor=AXIS_COLOR,
                tickfont=dict(color="white"),
                title=dict(font=dict(color="white")),
            ),
        ),
    )

    return fig