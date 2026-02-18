#!/usr/bin/env -S uv run
"""
OCR concurrency/load sweep for OpenAI-compatible vision endpoints.

Runs concurrent requests against one OCR image and writes machine-readable results
to JSON and CSV for graphing.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class SweepPoint:
    concurrency: int
    ok: int
    total: int
    errors: int
    wall_s: float
    req_s: float
    p50_s: float
    p95_s: float
    p99_s: float
    mean_s: float
    timeout_errors: int
    http_errors: int
    other_errors: int


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int((p / 100.0) * len(values)) - 1))
    return values[idx]


def _parse_concurrency_list(raw: str) -> list[int]:
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("No valid concurrency values provided")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR load/concurrency sweep")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument(
        "--concurrency",
        default="4,8,12,16,20,24,32,40,48,64",
        help="Comma-separated list, e.g. 4,8,16,32",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    url = f"{args.base_url.rstrip('/')}{args.endpoint}"
    conc_points = _parse_concurrency_list(args.concurrency)

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")

    def one_request() -> tuple[bool, float, str]:
        start = time.perf_counter()
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=args.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            ok = bool(content)
            err = ""
        except HTTPError as e:
            ok = False
            err = f"http:{e.code}"
        except URLError as e:
            ok = False
            err = f"url:{e.reason}"
        except TimeoutError:
            ok = False
            err = "timeout"
        except Exception as e:  # noqa: BLE001
            ok = False
            err = f"other:{e}"
        return ok, time.perf_counter() - start, err

    points: list[SweepPoint] = []
    run_started = time.time()

    for n in conc_points:
        print(f"Running concurrency={n} ...", flush=True)
        start = time.perf_counter()
        results: list[tuple[bool, float, str]] = []
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(one_request) for _ in range(n)]
            for fut in as_completed(futs):
                results.append(fut.result())
        wall = time.perf_counter() - start

        lats = [lat for _, lat, _ in results]
        ok_count = sum(1 for ok, _, _ in results if ok)
        errs = [err for ok, _, err in results if not ok]
        timeout_errors = sum(1 for e in errs if e.startswith("timeout"))
        http_errors = sum(1 for e in errs if e.startswith("http:"))
        other_errors = len(errs) - timeout_errors - http_errors

        point = SweepPoint(
            concurrency=n,
            ok=ok_count,
            total=len(results),
            errors=len(errs),
            wall_s=round(wall, 3),
            req_s=round(len(results) / wall, 3) if wall > 0 else 0.0,
            p50_s=round(_percentile(lats, 50), 3),
            p95_s=round(_percentile(lats, 95), 3),
            p99_s=round(_percentile(lats, 99), 3),
            mean_s=round(statistics.fmean(lats), 3),
            timeout_errors=timeout_errors,
            http_errors=http_errors,
            other_errors=other_errors,
        )
        points.append(point)
        print(asdict(point), flush=True)

    payload_out = {
        "meta": {
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "model": args.model,
            "image": str(image_path),
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "timeout_s": args.timeout_s,
            "concurrency_points": conc_points,
            "run_started_unix": run_started,
            "run_finished_unix": time.time(),
        },
        "results": [asdict(p) for p in points],
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload_out, indent=2))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concurrency",
                "ok",
                "total",
                "errors",
                "wall_s",
                "req_s",
                "p50_s",
                "p95_s",
                "p99_s",
                "mean_s",
                "timeout_errors",
                "http_errors",
                "other_errors",
            ],
        )
        writer.writeheader()
        for p in points:
            writer.writerow(asdict(p))

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
