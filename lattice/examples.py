EXAMPLES_BY_DIMENSION = {
    "2D": {
        "Square lattice": {
            "basis": "[[1, 0],\n [0, 1]]",
            "target": "[1.4, 0.7]",
            "cube_limit": 3,
        },
        "Skewed lattice": {
            "basis": "[[2, 1],\n [0, 1]]",
            "target": "[2.3, 1.7]",
            "cube_limit": 3,
        },
        "Already convenient basis": {
            "basis": "[[1, 1],\n [1, -1]]",
            "target": "[2.3, 1.7]",
            "cube_limit": 2,
        },
        "Intentionally bad basis": {
            "basis": "[[1, 8],\n [0, 1]]",
            "target": "[2.6, 1.3]",
            "cube_limit": 5,
        },
    },
    "3D": {
        "Simple cubic lattice": {
            "basis": "[[1, 0, 0],\n [0, 1, 0],\n [0, 0, 1]]",
            "target": "[1.4, 1.6, 0.8]",
            "cube_limit": 2,
        },
        "Skewed lattice": {
            "basis": "[[1, 0, 1],\n [0, 1, 1],\n [0, 0, 2]]",
            "target": "[1.4, 1.6, 2.2]",
            "cube_limit": 2,
        },
        "Less convenient basis": {
            "basis": "[[3, 1, 2],\n [1, 2, 1],\n [0, 1, 1]]",
            "target": "[2.2, 1.7, 1.3]",
            "cube_limit": 4,
        },
    },
}