#!/usr/bin/env python3
"""Measure cold-prompt streaming latency and server token throughput."""

import argparse
import json
import math
import time
import urllib.request


def request(url, payload):
    return urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18091")
    parser.add_argument("--prompts", default="128,512,1024,2048")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=128)
    args = parser.parse_args()
    lengths = [int(value) for value in args.prompts.split(",")]
    if min(lengths) <= 0 or args.repeats <= 0 or args.n_predict <= 0:
        parser.error("lengths and repetitions must be positive")
    text = "The service reads a record, checks its identifier, and returns the requested value. " * max(lengths)
    with urllib.request.urlopen(request(args.url + "/tokenize", {"content": text, "add_special": True}), timeout=60) as response:
        tokens = json.load(response)["tokens"]
    for length in lengths:
        if len(tokens) < length:
            raise RuntimeError("not enough prompt tokens")
        for repeat in range(args.repeats):
            payload = {"prompt": tokens[:length], "n_predict": args.n_predict, "temperature": 0, "seed": 123, "ignore_eos": True, "cache_prompt": False, "stream": True}
            started = time.monotonic()
            first = None
            final = None
            with urllib.request.urlopen(request(args.url + "/completion", payload), timeout=1800) as response:
                for raw in response:
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        continue
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    if event.get("content") and first is None:
                        first = time.monotonic() - started
                    if event.get("stop"):
                        final = event
            if first is None or final is None:
                raise RuntimeError("incomplete completion stream")
            timings = final["timings"]
            if timings["predicted_n"] != args.n_predict:
                raise RuntimeError("unexpected output token count")
            for key in ["prompt_per_second", "predicted_per_second"]:
                if not math.isfinite(timings[key]) or timings[key] <= 0:
                    raise RuntimeError("invalid throughput")
            print(json.dumps({"requested_prompt_tokens": length, "repeat": repeat + 1, "ttft_seconds": first, "elapsed_seconds": time.monotonic() - started, "timings": timings}), flush=True)


if __name__ == "__main__":
    main()
