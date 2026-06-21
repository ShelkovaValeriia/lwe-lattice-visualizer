import streamlit as st

from lattice.examples import EXAMPLES_BY_DIMENSION


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