"""Run the 5-arm x 3-seed modded-nanogpt FineWeb-10B probe in parallel on Modal L4 GPUs.

    modal run research/modal_speedrun_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import modal

ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6.0", "numpy", "huggingface_hub")
    .env({"TZ": "<-03>3"})
    .add_local_file(ROOT / "research" / "speedrun_probe.py", "/root/speedrun_probe.py")
)
vol = modal.Volume.from_name("teach-gpt-runs", create_if_missing=True)
app = modal.App("teach-gpt-speedrun-probe")

DATA_DIR = "/out/fineweb10B"


@app.function(image=image, volumes={"/out": vol}, timeout=600)
def ensure_fineweb_shards() -> str:
    """Download 1 FineWeb-10B train shard (~200MB) + 1 val shard (~20MB) once to Modal Volume."""
    from huggingface_hub import hf_hub_download

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    for fname in ("fineweb_val_000000.bin", "fineweb_train_000001.bin"):
        dst = Path(DATA_DIR) / fname
        if not dst.exists():
            print(f"Downloading {fname} from kjj0/fineweb10B-gpt2...", flush=True)
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=DATA_DIR,
            )
    vol.commit()
    return "ready"


@app.function(gpu="A10G", image=image, volumes={"/out": vol}, timeout=3600)
def run_probe(arm: str, seed: int, steps: int = 1500) -> dict:
    import torch

    print(f"[container] booting arm={arm} s{seed} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    out_path = f"/out/speedrun_probe/{arm}_s{seed}.json"
    cmd = [
        sys.executable,
        "/root/speedrun_probe.py",
        "--arm", arm,
        "--seed", str(seed),
        "--data_dir", DATA_DIR,
        "--steps", str(steps),
        "--out_json", out_path,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    res = None
    tail = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line_s = line.rstrip()
        tail.append(line_s)
        if len(tail) > 40:
            tail.pop(0)
        if line_s.startswith("RESULT_JSON "):
            res = json.loads(line_s[len("RESULT_JSON ") :])
        else:
            print(f"[{arm} s{seed}] {line_s}", flush=True)
    code = proc.wait()
    vol.commit()
    if res is None:
        return {"arm": arm, "seed": seed, "code": code, "log_tail": "\n".join(tail)}
    return res


@app.local_entrypoint()
def main(steps: int = 1500, seeds: int = 3, arms: str = "all"):
    print("Step 1: Ensuring FineWeb-10B shards exist on Modal Volume...", flush=True)
    ensure_fineweb_shards.remote()

    all_arms = ["base", "vssmax", "bssmax", "relu_fox", "bssmax_relu_fox"]
    run_arms = all_arms if arms == "all" else [a.strip() for a in arms.split(",") if a.strip()]
    jobs = [(arm, s, steps) for arm in run_arms for s in range(seeds)]
    print(f"Step 2: Launching {len(jobs)} parallel A10G probes ({run_arms} x {seeds} seeds, {steps} steps each)...", flush=True)

    results = []
    for r in run_probe.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(r, dict):
            results.append(r)
            if "val_loss_2048" in r:
                print(
                    f"DONE {r['arm']:15s} s{r['seed']} | {r['train_time_s']:5.1f}s ({r['ms_per_step']:5.1f}ms/st) "
                    f"| val2048={r['val_loss_2048']:.4f} | val4096={r['val_loss_4096']:.4f}",
                    flush=True,
                )
            else:
                print(f"FAIL {r}", flush=True)
        else:
            print(f"EXCEPTION {r!r}", flush=True)

    out_local = ROOT / "research" / "speedrun_probe_results.json"
    merged = {}
    if out_local.exists():
        for prev in json.loads(out_local.read_text()):
            merged[(prev["arm"], prev["seed"])] = prev
    for r in results:
        merged[(r["arm"], r["seed"])] = r
    arm_order = {a: i for i, a in enumerate(all_arms)}
    ordered = sorted(merged.values(), key=lambda x: (arm_order.get(x["arm"], 99), x["seed"]))
    out_local.write_text(json.dumps(ordered, indent=2))
    print(f"Saved {len(ordered)} merged probe results to {out_local}", flush=True)
