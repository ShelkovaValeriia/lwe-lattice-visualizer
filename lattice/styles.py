import streamlit as st


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