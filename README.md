# Lattice Visualizer

Educational Streamlit application for visualizing small 2D and 3D lattices.

## How to run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/lwe-lattice-visualizer.git
cd lwe-lattice-visualizer
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The app will open in the browser, usually at:

```text
http://localhost:8501
```

## About the project

This project is a small visualization tool for lattices used in the context of lattice-based cryptography and post-quantum cryptography.

The app shows how a lattice is generated from a basis matrix `B`.
Lattice points have the form:

```text
Bz
```

where `z` is an integer vector.

## Main functionality

The application supports:

* 2D and 3D lattice visualization;
* custom basis matrix input;
* prepared example lattices;
* visualization of basis vectors;
* visualization of the fundamental parallelogram / parallelepiped;
* numerical display of the fundamental area / volume;
* brute-force SVP search for small examples;
* brute-force CVP search for a selected target point;
* highlight modes for SVP, CVP and basis elements;
* comparison mode for comparing two different bases of the same lattice.

## Comparison mode

Comparison mode allows two bases to be shown side by side.

Both graphs use the same lattice points and the same scale.
Only the basis vectors and the fundamental domain change.

This makes it easier to compare convenient and inconvenient bases of the same lattice.

## Project structure

```text
lwe-lattice-visualizer/
├── app.py
├── requirements.txt
├── README.md
└── lattice/
    ├── algorithms.py
    ├── core.py
    ├── examples.py
    ├── plotting_2d.py
    ├── plotting_3d.py
    └── utils.py
```

## Note

SVP and CVP are computed by brute force only for small educational examples.
The project is intended for visualization and explanation, not for efficient cryptanalysis.
