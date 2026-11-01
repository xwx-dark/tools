"""Build a Postman JSON body for reviewing a phone-use alert image."""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2


def build_request_body(image_path: Path) -> dict[str, object]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    encoded, buffer = cv2.imencode(".jpg", image)
    if not encoded:
        raise ValueError(f"Could not encode image as JPEG: {image_path}")

    return {
        "job_id": "postman-phone-test",
        "camera_code": "test-camera",
        "events": [{
            "violation_type": "mobile_phone",
            "model_label": "Phone",
        }],
        "images": [base64.b64encode(buffer.tobytes()).decode("ascii")],
        "image_width": width,
        "image_height": height,
        "violation_detect": False,
        "violation_types": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a Postman JSON body to review a phone-use alert."
    )
    parser.add_argument("image", type=Path, help="Path to the image to review")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the JSON body to this file instead of stdout",
    )
    args = parser.parse_args()

    try:
        body = build_request_body(args.image)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    json_body = json.dumps(body, indent=2)
    if args.output:
        args.output.write_text(json_body + "\n", encoding="utf-8")
        print(f"Postman request body written to: {args.output}", file=sys.stderr)
    else:
        print(json_body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
