import math

import numpy as np
import plotly.graph_objects as go


PLOT_BG = "#0E1117"
GRID_COLOR = "rgba(180, 200, 255, 0.18)"
AXIS_COLOR = "rgba(255, 255, 255, 0.75)"
POINT_COLOR = "rgba(101, 184, 255, 0.85)"
BOUNDARY_COLOR = "rgba(255, 196, 0, 0.55)"


def _residue_axis_bounds(
    modulus: int,
    centered: bool,
) -> tuple[int, int]:
    if centered:
        return -(modulus // 2), (modulus - 1) // 2

    return 0, modulus - 1


def _tick_values(lower: int, upper: int) -> list[int]:
    count = upper - lower + 1

    if count <= 15:
        return list(range(lower, upper + 1))

    step = max(1, math.ceil(count / 10))
    ticks = list(range(lower, upper + 1, step))

    if ticks[-1] != upper:
        ticks.append(upper)

    return ticks


def _add_modular_boundary_box(
    fig: go.Figure,
    lower: int,
    upper: int,
) -> None:
    """
    Draw the boundary of one displayed modulo-q cell.

    Residues lie on integer coordinates, therefore the geometric cell
    boundaries are half a unit outside the smallest and largest residues.
    """
    low = lower - 0.5
    high = upper + 0.5

    vertices = np.array(
        [
            [low, low, low],
            [high, low, low],
            [low, high, low],
            [high, high, low],
            [low, low, high],
            [high, low, high],
            [low, high, high],
            [high, high, high],
        ],
        dtype=float,
    )

    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    x_values = []
    y_values = []
    z_values = []

    for start_index, end_index in edges:
        start = vertices[start_index]
        end = vertices[end_index]

        x_values.extend([start[0], end[0], None])
        y_values.extend([start[1], end[1], None])
        z_values.extend([start[2], end[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            name="Modulo-q cell",
            line=dict(
                color=BOUNDARY_COLOR,
                width=3,
                dash="dot",
            ),
            hoverinfo="skip",
        )
    )


def create_modular_3d_plot(
    display_points: np.ndarray,
    standard_points: np.ndarray,
    representative_coefficients: np.ndarray,
    multiplicities: np.ndarray,
    modulus: int,
    centered: bool,
) -> go.Figure:
    """Create the 3D visualization of unique points Bz modulo q."""
    lower, upper = _residue_axis_bounds(modulus, centered)
    tick_values = _tick_values(lower, upper)

    representation_name = "Centered" if centered else "Standard"

    hover_text = [
        (
            f"Displayed residue: {display.tolist()}"
            f"<br>Standard residue: {standard.tolist()}"
            f"<br>Example z: {coefficient.tolist()}"
            f"<br>Mappings to this residue: {int(multiplicity)}"
        )
        for display, standard, coefficient, multiplicity in zip(
            display_points,
            standard_points,
            representative_coefficients,
            multiplicities,
        )
    ]

    marker_sizes = 5 + np.minimum(multiplicities - 1, 6)

    fig = go.Figure()

    _add_modular_boundary_box(
        fig=fig,
        lower=lower,
        upper=upper,
    )

    fig.add_trace(
        go.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode="markers",
            name="Unique residues",
            marker=dict(
                size=marker_sizes,
                color=POINT_COLOR,
                opacity=0.88,
                line=dict(
                    width=0.7,
                    color="rgba(255, 255, 255, 0.30)",
                ),
            ),
            text=hover_text,
            hoverinfo="text",
        )
    )

    cell_low = lower - 0.5
    cell_high = upper + 0.5
    axis_range = [cell_low - 0.15, cell_high + 0.15]

    common_axis = dict(
        range=axis_range,
        tickmode="array",
        tickvals=tick_values,
        backgroundcolor=PLOT_BG,
        gridcolor=GRID_COLOR,
        zerolinecolor=AXIS_COLOR,
        linecolor=AXIS_COLOR,
        tickfont=dict(color="white"),
        title=dict(font=dict(color="white")),
    )

    fig.update_layout(
        title=dict(
            text=(
                f"3D Modular Lattice: Bz mod {modulus} "
                f"({representation_name} residues)"
            ),
            font=dict(color="white", size=18),
        ),
        height=760,
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
            xaxis_title="x₁",
            yaxis_title="x₂",
            zaxis_title="x₃",
            aspectmode="cube",
            xaxis=common_axis,
            yaxis=common_axis,
            zaxis=common_axis,
        ),
        margin=dict(l=0, r=0, b=0, t=55),
    )

    return fig