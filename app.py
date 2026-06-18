import ast
import itertools
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


EPS = 1e-9


def parse_matrix(text: str) -> np.ndarray:
    """
    Parses a matrix from text input.

    Example:
    [[2, 1],
     [0, 1]]
    """
    try:
        data = ast.literal_eval(text)
        matrix = np.array(data, dtype=float)
    except Exception as exc:
        raise ValueError("Matrix must be written as a valid Python list, for example [[2, 1], [0, 1]].") from exc

    if matrix.ndim != 2:
        raise ValueError("Basis matrix must be two-dimensional.")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Basis matrix must be square, for example 2x2 or 3x3.")

    det = np.linalg.det(matrix)
    if abs(det) < EPS:
        raise ValueError("Basis matrix is singular. Its determinant is 0, so it does not define a full-rank lattice.")

    return matrix


def parse_vector(text: str, dimension: int) -> np.ndarray:
    """
    Parses target vector from text input.

    Example:
    [2.3, 1.7]
    """
    try:
        data = ast.literal_eval(text)
        vector = np.array(data, dtype=float)
    except Exception as exc:
        raise ValueError("Target point must be written as a valid Python list, for example [2.3, 1.7].") from exc

    if vector.ndim != 1:
        raise ValueError("Target point must be a vector, for example [2.3, 1.7].")

    if len(vector) != dimension:
        raise ValueError(f"Target point must have dimension {dimension}.")

    return vector


def generate_lattice_points(B: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates lattice points of the form v = Bz,
    where z is an integer vector with coordinates in [-radius, radius].

    Basis vectors are stored as columns of B.
    """
    n = B.shape[0]

    coefficient_vectors = np.array(
        list(itertools.product(range(-radius, radius + 1), repeat=n)),
        dtype=int
    )

    lattice_points = coefficient_vectors @ B.T

    return coefficient_vectors, lattice_points


def find_shortest_vector(
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Finds the shortest non-zero lattice vector by brute force.
    """
    non_zero_mask = np.any(coefficient_vectors != 0, axis=1)

    non_zero_coefficients = coefficient_vectors[non_zero_mask]
    non_zero_points = lattice_points[non_zero_mask]

    norms = np.linalg.norm(non_zero_points, axis=1)
    min_index = int(np.argmin(norms))

    return (
        non_zero_coefficients[min_index],
        non_zero_points[min_index],
        float(norms[min_index])
    )


def find_closest_point(
    target: np.ndarray,
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solves CVP approximately by brute force inside the selected range.
    """
    distances = np.linalg.norm(lattice_points - target, axis=1)
    min_index = int(np.argmin(distances))

    return (
        coefficient_vectors[min_index],
        lattice_points[min_index],
        float(distances[min_index])
    )


def gauss_reduce_2d(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs simple Gaussian lattice basis reduction in 2D.

    The new basis B_reduced defines the same lattice:
    B_reduced = B * U,
    where U is an integer unimodular matrix with det(U) = ±1.
    """
    if B.shape != (2, 2):
        raise ValueError("Gauss reduction is implemented only for 2D bases.")

    B_reduced = B.copy().astype(float)
    U = np.eye(2, dtype=int)

    changed = True
    while changed:
        changed = False

        b1 = B_reduced[:, 0]
        b2 = B_reduced[:, 1]

        if np.dot(b2, b2) < np.dot(b1, b1):
            B_reduced[:, [0, 1]] = B_reduced[:, [1, 0]]
            U[:, [0, 1]] = U[:, [1, 0]]
            changed = True
            continue

        b1 = B_reduced[:, 0]
        b2 = B_reduced[:, 1]

        mu = int(round(np.dot(b1, b2) / np.dot(b1, b1)))

        if mu != 0:
            B_reduced[:, 1] = B_reduced[:, 1] - mu * B_reduced[:, 0]
            U[:, 1] = U[:, 1] - mu * U[:, 0]
            changed = True

    return B_reduced, U


def format_array(array: np.ndarray, precision: int = 4) -> str:
    """
    Formats numpy arrays for nicer output.
    """
    rounded = np.round(array.astype(float), precision)
    return str(rounded.tolist())


def create_2d_plot(
    B: np.ndarray,
    coefficient_vectors: np.ndarray,
    lattice_points: np.ndarray,
    shortest_vector: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    closest_point: Optional[np.ndarray] = None,
    reduced_basis: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Creates a 2D Plotly visualization of the lattice.
    """
    fig = go.Figure()

    x = lattice_points[:, 0]
    y = lattice_points[:, 1]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Lattice points",
            marker=dict(size=7, color="lightgray", line=dict(width=1, color="gray")),
            text=[f"z = {z.tolist()}<br>Bz = {np.round(p, 4).tolist()}" for z, p in zip(coefficient_vectors, lattice_points)],
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

    basis_colors = ["blue", "green"]
    basis_names = ["b₁", "b₂"]

    for i, basis_vector in enumerate([b1, b2]):
        fig.add_trace(
            go.Scatter(
                x=[0, basis_vector[0]],
                y=[0, basis_vector[1]],
                mode="lines+markers+text",
                name=f"Basis vector {basis_names[i]}",
                line=dict(color=basis_colors[i], width=4),
                marker=dict(size=8, color=basis_colors[i]),
                text=["", basis_names[i]],
                textposition="top center",
            )
        )

    if reduced_basis is not None:
        r1 = reduced_basis[:, 0]
        r2 = reduced_basis[:, 1]

        for i, reduced_vector in enumerate([r1, r2]):
            fig.add_trace(
                go.Scatter(
                    x=[0, reduced_vector[0]],
                    y=[0, reduced_vector[1]],
                    mode="lines+markers+text",
                    name=f"Reduced basis vector r{i + 1}",
                    line=dict(color="purple", width=3, dash="dash"),
                    marker=dict(size=8, color="purple"),
                    text=["", f"r{i + 1}"],
                    textposition="bottom center",
                )
            )

    if shortest_vector is not None:
        fig.add_trace(
            go.Scatter(
                x=[0, shortest_vector[0]],
                y=[0, shortest_vector[1]],
                mode="lines+markers+text",
                name="Shortest vector",
                line=dict(color="red", width=5),
                marker=dict(size=10, color="red"),
                text=["", "SVP"],
                textposition="top center",
            )
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
        title="2D Lattice Visualization",
        xaxis_title="x",
        yaxis_title="y",
        width=900,
        height=700,
        showlegend=True,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
    )

    return fig


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
    det_B = float(np.linalg.det(B))
    lattice_determinant = abs(det_B)

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
        st.subheader("Results")

        st.metric("Dimension", n)
        st.metric("Number of lattice points", total_points)
        st.metric("|det(B)|", round(lattice_determinant, 6))

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
            )
            fig.update_layout(title="2D Projection of the Lattice")
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