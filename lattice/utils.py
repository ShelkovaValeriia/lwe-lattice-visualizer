import numpy as np


def format_array(array: np.ndarray, precision: int = 4) -> str:
    """
    Formats numpy arrays for nicer output in Streamlit.
    """
    rounded = np.round(array.astype(float), precision)
    return str(rounded.tolist())