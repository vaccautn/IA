"""Single source of truth for the integer BCS domain and image set."""

BCS_DOMAIN_ID = "bcs-integer-1-5"
BCS_CLASS_SCORES = (1, 2, 3, 4, 5)
CLASS_NAMES = tuple(str(score) for score in BCS_CLASS_SCORES)
SCORE_MIN = BCS_CLASS_SCORES[0]
SCORE_MAX = BCS_CLASS_SCORES[-1]
SCORE_BASE = SCORE_MIN
SCORE_STEP = 1
NUM_CLASSES = len(BCS_CLASS_SCORES)
NUM_THRESHOLDS = NUM_CLASSES - 1

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
