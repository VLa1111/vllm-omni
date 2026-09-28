# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture one image from a running vLLM-Omni server, for the quality comparison.

`run_e2e.sh` calls this once per batch-1 config with the same prompt, seed and
step count as the benchmark, so `collect.py` can report the PSNR of every config
against the baseline image.

Standalone::

    python capture_image.py --url http://127.0.0.1:8099 --size 1536 --out img.png
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = "A stylish woman riding a motorcycle in NYC, movie poster style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Server base URL, e.g. http://127.0.0.1:8099")
    parser.add_argument("--out", required=True, help="Where to write the PNG")
    parser.add_argument("--size", type=int, default=1536, help="Square output size")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=9.0, help="text_guidance_scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=None, help="Served model name, if the server needs one")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Per-request timeout in seconds")
    return parser.parse_args()


def extract_base64(payload: dict) -> str:
    """Pull the first base64 image out of a chat-completions or images response."""
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("b64_json"):
        return data[0]["b64_json"]
    choices = payload.get("choices") or []
    if not choices:
        raise SystemExit(f"no choices in the response: {json.dumps(payload)[:500]}")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, list):
        for part in content:
            url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) else None
            if url:
                return url.split(",", 1)[-1]
    if isinstance(content, str) and "base64," in content:
        return content.split("base64,", 1)[-1].rstrip(") \n")
    raise SystemExit(f"no image in the response: {json.dumps(choices[0])[:500]}")


def main() -> None:
    args = parse_args()
    body = {
        "messages": [{"role": "user", "content": args.prompt}],
        "extra_body": {
            "height": args.size,
            "width": args.size,
            "num_inference_steps": args.steps,
            "text_guidance_scale": args.guidance,
            "seed": args.seed,
        },
    }
    if args.model:
        body["model"] = args.model
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} from the server: {exc.read()[:500].decode(errors='replace')}") from exc
    image = base64.b64decode(extract_base64(payload))
    Path(args.out).write_bytes(image)
    print(f"wrote {args.out} ({len(image)} bytes) in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
