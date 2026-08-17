import streamlit as st

from lattice.examples import EXAMPLES_BY_DIMENSION
from lattice.modular import is_prime, nearest_primes, recommended_prime


VISUALIZATION_MODES = [
    "Classical lattice",
    "Modular lattice",
]

RESIDUE_REPRESENTATIONS = [
    "Centered",
    "Standard",
]


def _set_modulus(value: int) -> None:
    """Update the prime modulus input from a suggestion button."""
    st.session_state["modulus_q"] = int(value)


def show_visualization_mode() -> str:
    """Show the top-level visualization mode selector."""
    with st.sidebar:
        st.header("Mode")

        visualization_mode = st.radio(
            "Visualization mode",
            options=VISUALIZATION_MODES,
            index=0,
            help=(
                "Classical lattice keeps the existing SVP/CVP visualizer. "
                "Modular lattice introduces integer arithmetic modulo a prime q."
            ),
        )

    return visualization_mode


def show_modular_sidebar() -> tuple[str, int, str, int]:
    """Show controls for the modular 3D visualization mode."""
    with st.sidebar:
        st.header("Modular input")

        st.selectbox(
            "Dimension",
            options=["3D"],
            index=0,
            disabled=True,
            help="The first modular visualization is intentionally restricted to 3D.",
        )

        examples_for_dimension = EXAMPLES_BY_DIMENSION["3D"]

        selected_example_name = st.selectbox(
            "Preset basis",
            options=list(examples_for_dimension.keys()),
            index=1,
            help=(
                "Choose a 3D integer basis. The skewed preset is useful for "
                "seeing wrap-around modulo q."
            ),
        )

        selected_example = examples_for_dimension[selected_example_name]

        basis_text = st.text_area(
            "Integer basis matrix B",
            value=selected_example["basis"],
            height=130,
            help="For modular mode, B must be a 3×3 matrix with integer entries.",
        )

        st.divider()
        st.subheader("Modulo q")

        if "modulus_q" not in st.session_state:
            st.session_state["modulus_q"] = 7

        modulus = int(
            st.number_input(
                "Prime modulus q",
                min_value=2,
                step=1,
                key="modulus_q",
                help="For this visualization, q is restricted to prime numbers.",
            )
        )

        if is_prime(modulus):
            st.success(f"q = {modulus} is prime.")
        else:
            lower_prime, upper_prime = nearest_primes(modulus)
            recommendation = recommended_prime(modulus)

            st.error(f"q = {modulus} is not prime.")

            if recommendation is not None:
                st.caption(f"Recommended nearest prime: q = {recommendation}")
            else:
                st.caption("Both neighboring primes are equally close.")

            button_columns = st.columns(2)

            if lower_prime is not None:
                with button_columns[0]:
                    st.button(
                        f"Use {lower_prime}",
                        key=f"use_lower_prime_{lower_prime}",
                        on_click=_set_modulus,
                        args=(lower_prime,),
                        use_container_width=True,
                    )

            with button_columns[1]:
                st.button(
                    f"Use {upper_prime}",
                    key=f"use_upper_prime_{upper_prime}",
                    on_click=_set_modulus,
                    args=(upper_prime,),
                    use_container_width=True,
                )

        residue_representation = st.radio(
            "Residue representation",
            options=RESIDUE_REPRESENTATIONS,
            index=0,
            help=(
                "Centered representation places residues around zero. "
                "Standard representation uses values from 0 to q - 1."
            ),
        )

        if residue_representation == "Centered":
            lower = -(modulus // 2)
            upper = (modulus - 1) // 2
            st.caption(f"Displayed residues: [{lower}, ..., {upper}]")
        else:
            st.caption(f"Displayed residues: [0, ..., {modulus - 1}]")

        st.divider()
        st.subheader("Coefficient vectors")

        coefficient_limit = st.slider(
            "Coefficient range r",
            min_value=1,
            max_value=5,
            value=2,
            help=(
                "The app generates all integer vectors z in [-r, r]³ "
                "before applying Bz mod q."
            ),
        )

        coefficient_count = (2 * coefficient_limit + 1) ** 3
        st.caption(
            f"Generated z vectors: (2·{coefficient_limit}+1)³ = "
            f"{coefficient_count}"
        )

    return (
        basis_text,
        modulus,
        residue_representation,
        coefficient_limit,
    )


def show_sidebar() -> tuple[str, dict, str, int, bool, str, str]:
    """
    Show all input controls in the sidebar for the classical mode.
    """
    with st.sidebar:
        st.header("Classical input")

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