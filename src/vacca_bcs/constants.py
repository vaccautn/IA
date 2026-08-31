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

# Image suffixes accepted by the integer dataset loader.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = ("train", "val")
MANIFEST_FILENAME = "manifest.json"
