# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai==1.61.0",
#     "python-dotenv==1.0.1",
#     "tqdm",
#     "pillow",
# ]
# ///
"""
OCRBench evaluation for Google Gemini models (VQA tasks)
Uses the modern google-genai SDK with vision support.

Usage:
    uv run eval_gemini.py
    uv run eval_gemini.py --model gemini-2.5-flash
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv()


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/png")


def run_inference(client: genai.Client, model: str, image_path: str, question: str) -> str:
    """Run VQA inference using Gemini vision API."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_type = get_image_mime_type(image_path)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            question,
        ],
    )
    return response.text.strip()


def evaluate_prediction(predict: str, answers: str | list, dataset_name: str) -> int:
    """Evaluate if prediction matches any of the answers."""
    if dataset_name == "HME100k":
        predict_clean = predict.strip().replace("\n", " ").replace(" ", "")
        if isinstance(answers, list):
            for answer in answers:
                answer_clean = answer.strip().replace("\n", " ").replace(" ", "")
                if answer_clean in predict_clean:
                    return 1
        else:
            answer_clean = answers.strip().replace("\n", " ").replace(" ", "")
            if answer_clean in predict_clean:
                return 1
    else:
        predict_clean = predict.lower().strip().replace("\n", " ")
        if isinstance(answers, list):
            for answer in answers:
                answer_clean = answer.lower().strip().replace("\n", " ")
                if answer_clean in predict_clean:
                    return 1
        else:
            answer_clean = answers.lower().strip().replace("\n", " ")
            if answer_clean in predict_clean:
                return 1
    return 0


def print_ocrbench_results(data: list) -> None:
    """Print OCRBench-style results breakdown."""
    scores = {
        "Regular Text Recognition": 0,
        "Irregular Text Recognition": 0,
        "Artistic Text Recognition": 0,
        "Handwriting Recognition": 0,
        "Digit String Recognition": 0,
        "Non-Semantic Text Recognition": 0,
        "Scene Text-centric VQA": 0,
        "Doc-oriented VQA": 0,
        "Key Information Extraction": 0,
        "Handwritten Mathematical Expression Recognition": 0,
    }

    for item in data:
        if "result" in item and item["type"] in scores:
            scores[item["type"]] += item["result"]

    recognition_score = sum(
        scores[k]
        for k in [
            "Regular Text Recognition",
            "Irregular Text Recognition",
            "Artistic Text Recognition",
            "Handwriting Recognition",
            "Digit String Recognition",
            "Non-Semantic Text Recognition",
        ]
    )
    final_score = recognition_score + sum(
        scores[k]
        for k in [
            "Scene Text-centric VQA",
            "Doc-oriented VQA",
            "Key Information Extraction",
            "Handwritten Mathematical Expression Recognition",
        ]
    )

    print("\n" + "=" * 60)
    print("OCRBench Results")
    print("=" * 60)
    print(f"Text Recognition (Total 300): {recognition_score}")
    print("-" * 60)
    print(f"  Regular Text Recognition (50): {scores['Regular Text Recognition']}")
    print(f"  Irregular Text Recognition (50): {scores['Irregular Text Recognition']}")
    print(f"  Artistic Text Recognition (50): {scores['Artistic Text Recognition']}")
    print(f"  Handwriting Recognition (50): {scores['Handwriting Recognition']}")
    print(f"  Digit String Recognition (50): {scores['Digit String Recognition']}")
    print(f"  Non-Semantic Text Recognition (50): {scores['Non-Semantic Text Recognition']}")
    print("-" * 60)
    print(f"Scene Text-centric VQA (200): {scores['Scene Text-centric VQA']}")
    print(f"Doc-oriented VQA (200): {scores['Doc-oriented VQA']}")
    print(f"Key Information Extraction (200): {scores['Key Information Extraction']}")
    print(f"Handwritten Math Expression (100): {scores['Handwritten Mathematical Expression Recognition']}")
    print("=" * 60)
    print(f"FINAL SCORE (Total 1000): {final_score}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="OCRBench evaluation for Google Gemini")
    parser.add_argument("--api-key", type=str, default=os.environ.get("GEMINI_API_KEY"), help="Google API key")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="Model name")
    parser.add_argument("--image-folder", type=str, default="./OCRBench_Images", help="Path to images")
    parser.add_argument("--benchmark-file", type=str, default="./OCRBench/OCRBench.json", help="Benchmark JSON")
    parser.add_argument("--output", type=str, default=None, help="Output file (auto-generated if not specified)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key required. Set GEMINI_API_KEY in .env or use --api-key")

    # Auto-generate output filename from model
    if args.output is None:
        model_safe = args.model.replace("/", "_").replace("-", "_")
        args.output = f"./results/{model_safe}.json"

    client = genai.Client(api_key=args.api_key)

    with open(args.benchmark_file) as f:
        data = json.load(f)

    if args.limit:
        data = data[: args.limit]

    completed_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
            for item in existing:
                if "predict" in item:
                    completed_ids.add(item["id"])
                    for d in data:
                        if d["id"] == item["id"]:
                            d["predict"] = item["predict"]
                            d["result"] = item.get("result", 0)
        print(f"Resuming: {len(completed_ids)} items already completed")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Running evaluation on {len(data)} samples with model {args.model}...")
    for item in tqdm(data):
        if item["id"] in completed_ids:
            continue

        image_path = os.path.join(args.image_folder, item["image_path"])
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            predict = run_inference(client, args.model, image_path, item["question"])
            item["predict"] = predict
            item["result"] = evaluate_prediction(predict, item["answers"], item["dataset_name"])
        except Exception as e:
            print(f"Error on {item['id']}: {e}")
            item["predict"] = ""
            item["result"] = 0

        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)

    if len(data) == 1000:
        print_ocrbench_results(data)
    else:
        print(f"\nCompleted {len(data)} samples")
        correct = sum(1 for d in data if d.get("result", 0) == 1)
        print(f"Accuracy: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")


if __name__ == "__main__":
    main()
