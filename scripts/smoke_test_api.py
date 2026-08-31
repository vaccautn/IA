"""Quick direct detector/schema check (does not start FastAPI or make HTTP calls)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_api.detection import get_detector  # noqa: E402
from vacca_api.schemas import DetectResponse, HealthResponse  # noqa: E402

MODEL_PATH = ROOT / "outputs" / "training" / "combined-v2-finetune" / "weights" / "best.pt"

# Test image
test_img = ROOT / "data" / "cow-detection-navids" / "valid" / "images"
images = list(test_img.glob("*.jpg"))
if not images:
    print("[FAIL] No test images found")
    sys.exit(1)

test_file = images[0]
print(f"Test image: {test_file.name}")

# 1. Direct health schema construction
detector = get_detector(model_path=MODEL_PATH)
health = HealthResponse(
    model_loaded=True,
    model_path=detector.model_path,
    gpu_available=detector.gpu_available,
)
print(f"\n[HEALTH] {health.model_dump_json(indent=2)}")

# 2. Detection
image_bytes = test_file.read_bytes()
detections, img_w, img_h, elapsed_ms = detector.detect(image_bytes)

resp = DetectResponse(
    cow_detected=len(detections) > 0,
    detection_count=len(detections),
    detections=detections,
    image_width=img_w,
    image_height=img_h,
    inference_time_ms=round(elapsed_ms, 2),
)
print(f"\n[DETECT] {resp.model_dump_json(indent=2)}")

print("\n[OK] Direct detector/schema check completed")
