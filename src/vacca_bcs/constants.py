"""Single source of truth for the ordinal BCS class scale and image set."""

# Ordered BCS class labels; index i corresponds to score SCORE_BASE + i * SCORE_STEP.
CLASS_NAMES = ["3.25", "3.5", "3.75", "4.0", "4.25"]
SCORE_BASE = 3.25
SCORE_STEP = 0.25

# Image suffixes accepted by the dataset loader and the dataset builder.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CHUNK_SIZE = 1 << 20
DEFAULT_MAX_PER_CLASS = 6000
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.2
SPLITS = ("train", "val")
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
BACKUP_SUFFIX = ".backup-recovery"
