"""Single source of truth for the BCS category 1..5 domain and image set."""

BCS_DOMAIN_ID = "bcs-category-1-5-v1"
CHECKPOINT_SCHEMA_VERSION = "bcs-category-coral-checkpoint-v1"
BCS_CLASS_SCORES = (1, 2, 3, 4, 5)
CLASS_NAMES = tuple(str(score) for score in BCS_CLASS_SCORES)
SCORE_MIN = BCS_CLASS_SCORES[0]
SCORE_MAX = BCS_CLASS_SCORES[-1]
SCORE_BASE = SCORE_MIN
SCORE_STEP = 1
NUM_CLASSES = len(BCS_CLASS_SCORES)
NUM_THRESHOLDS = NUM_CLASSES - 1

# Image suffixes accepted by the category dataset loader.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = ("train", "val", "test")
MANIFEST_FILENAME = "manifest.json"
