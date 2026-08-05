#!/usr/bin/env python3
"""
Qwen3-VL performs only tasks it handles reliably:
- target detection
- bounding box
- simple visual properties

It does NOT predict the final grasp pixel.
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import cv2
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def base_dir():
    return Path(__file__).resolve().parent


def parse_args():
    base = base_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(base / "data" / "captured_rgb.jpg"))
    parser.add_argument("--target", default="")
    parser.add_argument("--model", default="/home/abr-lab/vlm/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", default=str(base / "results"))
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


def parse_json(text):
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip())
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    options = [cleaned] + ([match.group(0)] if match else [])

    for option in options:
        try:
            value = json.loads(option)
            if isinstance(value, dict):
                return value
        except Exception:
            try:
                value = ast.literal_eval(option)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    raise RuntimeError("Could not parse JSON from Qwen output")


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    args = parse_args()
    if not args.target.strip():
        args.target = input("Which object should the robot find? ").strip()
    if not args.target:
        raise RuntimeError("Target cannot be empty")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    height, width = image.shape[:2]

    print(f"Loading model: {args.model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
    )

    prompt = f"""
Find exactly one target object: "{args.target}".

Return one compact JSON object only:
{{
  "object": "{args.target}",
  "bbox_1000": [x1, y1, x2, y2],
  "category": "maximum 3 words",
  "shape": "maximum 5 words",
  "hollow": true_or_false_or_null,
  "opening_visible": true_or_false_or_null,
  "handle_visible": true_or_false_or_null,
  "rigid": true_or_false_or_null,
  "deformable": true_or_false_or_null
}}

Rules:
- bbox_1000 tightly surrounds only the target.
- Select one value for each property.
- Do not predict a grasp point.
- Do not output markdown or explanation.
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": args.image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"Detecting target and properties: {args.target}")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )

    trimmed = [
        output[len(input_ids):]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    print("Raw Qwen output:")
    print(raw)
    parsed = parse_json(raw)

    raw_bbox = parsed.get("bbox_1000")
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise RuntimeError("Qwen did not return bbox_1000")

    x1 = round(float(raw_bbox[0]) * width / 1000.0)
    y1 = round(float(raw_bbox[1]) * height / 1000.0)
    x2 = round(float(raw_bbox[2]) * width / 1000.0)
    y2 = round(float(raw_bbox[3]) * height / 1000.0)

    x1, x2 = sorted((clamp(x1, 0, width - 1), clamp(x2, 0, width - 1)))
    y1, y2 = sorted((clamp(y1, 0, height - 1), clamp(y2, 0, height - 1)))

    result = {
        "success": True,
        "target": args.target,
        "object": str(parsed.get("object", args.target)),
        "image": str(Path(args.image).resolve()),
        "image_size": [width, height],
        "bbox": [x1, y1, x2, y2],
        "category": str(parsed.get("category", "unknown")),
        "shape": str(parsed.get("shape", "unknown")),
        "properties": {
            "hollow": parsed.get("hollow"),
            "opening_visible": parsed.get("opening_visible"),
            "handle_visible": parsed.get("handle_visible"),
            "rigid": parsed.get("rigid"),
            "deformable": parsed.get("deformable"),
        },
        "raw_output": raw,
    }

    json_path = output_dir / "object_analysis.json"
    json_path.write_text(json.dumps(result, indent=2))

    annotated = image.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
    cv2.putText(
        annotated,
        f"{args.target}: {result['category']}, {result['shape']}",
        (max(10, x1), max(32, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    image_path = output_dir / "object_analysis.jpg"
    cv2.imwrite(str(image_path), annotated)

    print(json.dumps(result, indent=2))
    print(f"Object JSON: {json_path}")
    print(f"Object image: {image_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
