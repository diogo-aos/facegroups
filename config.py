from pathlib import Path

FACE_CONFIDENCE_THRESH = 0.95

src_dirs = [
    Path("/some/path/facerec"),
]

# files
database_path = Path("database")

VALID_IMG_EXT = [
    "jpg",
    "JPG",
    "jpeg",
    "JPEG",
]

# clustering

CLUSTER_MIN_IMG = 2
N_CLUSTERS = 200
MAX_CLUSTER_SIZE = 300


# # # VIZ
GRID_SIZE = (8, 20) # width, height
