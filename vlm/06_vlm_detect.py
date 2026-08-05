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
    parser.add_argument("--max-new-tokens", type=int, default=320)
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

The robot uses a parallel two-finger gripper with a maximum opening of 0.10 m.
Infer an object-specific grasp strategy from only the visible image.

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
  "deformable": true_or_false_or_null,
  "grasp_strategy": "one enum value",
  "preferred_grasp_region": "maximum 8 words",
  "grasp_region_bbox_1000": [x1, y1, x2, y2],
  "gripper_center_rule": "one enum value",
  "grasp_axis_rule": "one enum value",
  "avoid_grasp_regions": ["maximum 5 words", "maximum 5 words"],
  "grasp_confidence": 0.0_to_1.0,
  "grasp_reason": "maximum 20 words"
}}

Allowed grasp_strategy values:
- body_center
- handle_pinch
- neck_pinch
- outer_side_pinch
- rim_pinch
- edge_pinch
- top_surface
- not_graspable

Allowed gripper_center_rule values:
- use_object_center
- use_region_center
- shift_outward
- shift_inward
- not_applicable

Allowed grasp_axis_rule values:
- minor_axis
- radial
- perpendicular_to_handle
- parallel_to_edge
- unknown

Rules:
- bbox_1000 tightly surrounds only the target.
- Consider object shape, visible opening, rigidity, deformability, handles, narrow regions, and free space for two fingers.
- Prefer rigid, stable, narrow, pinchable regions.
- Prefer handles, stems, necks, side walls, or stable edges when appropriate.
- grasp_region_bbox_1000 must tightly surround only the preferred graspable sub-region, not the whole object.
- For handle_pinch, grasp_region_bbox_1000 must surround only the rigid handle and exclude bristles or the brush head.
- For body_center, grasp_region_bbox_1000 may cover the central graspable body.
- For an open cup, bowl, or container, use rim_pinch or outer_side_pinch only when the opening is clearly visible.
- Use shift_outward only for a visible open/hollow rim grasp where the gripper center must stay outside the opening.
- Never use shift_outward for a solid object.
- Avoid bristles, soft parts, fragile parts, sharp parts, open empty centers, and heavily occluded regions.
- If a safe grasp is not visually supported, return not_graspable.
- Do not predict an exact grasp pixel.
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

    raw_grasp_bbox = parsed.get("grasp_region_bbox_1000")
    grasp_region_bbox = None
    if isinstance(raw_grasp_bbox, list) and len(raw_grasp_bbox) == 4:
        gx1 = round(float(raw_grasp_bbox[0]) * width / 1000.0)
        gy1 = round(float(raw_grasp_bbox[1]) * height / 1000.0)
        gx2 = round(float(raw_grasp_bbox[2]) * width / 1000.0)
        gy2 = round(float(raw_grasp_bbox[3]) * height / 1000.0)

        gx1, gx2 = sorted((
            clamp(gx1, x1, x2),
            clamp(gx2, x1, x2),
        ))
        gy1, gy2 = sorted((
            clamp(gy1, y1, y2),
            clamp(gy2, y1, y2),
        ))

        if gx2 - gx1 >= 4 and gy2 - gy1 >= 4:
            grasp_region_bbox = [gx1, gy1, gx2, gy2]

    allowed_strategies = {
        "body_center",
        "handle_pinch",
        "neck_pinch",
        "outer_side_pinch",
        "rim_pinch",
        "edge_pinch",
        "top_surface",
        "not_graspable",
    }
    allowed_center_rules = {
        "use_object_center",
        "use_region_center",
        "shift_outward",
        "shift_inward",
        "not_applicable",
    }
    allowed_axis_rules = {
        "minor_axis",
        "radial",
        "perpendicular_to_handle",
        "parallel_to_edge",
        "unknown",
    }

    strategy = str(parsed.get("grasp_strategy", "not_graspable"))
    center_rule = str(parsed.get("gripper_center_rule", "not_applicable"))
    axis_rule = str(parsed.get("grasp_axis_rule", "unknown"))

    if strategy not in allowed_strategies:
        strategy = "not_graspable"
    if center_rule not in allowed_center_rules:
        center_rule = "not_applicable"
    if axis_rule not in allowed_axis_rules:
        axis_rule = "unknown"

    # Safety consistency correction: outward shifting is allowed only for
    # a visible open/hollow rim grasp.
    hollow = parsed.get("hollow")
    opening_visible = parsed.get("opening_visible")
    if center_rule == "shift_outward":
        if not (
            strategy == "rim_pinch"
            and hollow is True
            and opening_visible is True
        ):
            center_rule = "use_region_center"

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
        "grasp_affordance": {
            "strategy": strategy,
            "preferred_region": str(
                parsed.get("preferred_grasp_region", "unknown")
            ),
            "region_bbox": grasp_region_bbox,
            "gripper_center_rule": center_rule,
            "grasp_axis_rule": axis_rule,
            "avoid_regions": parsed.get("avoid_grasp_regions", []),
            "confidence": float(parsed.get("grasp_confidence", 0.0) or 0.0),
            "reason": str(parsed.get("grasp_reason", "unknown")),
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
