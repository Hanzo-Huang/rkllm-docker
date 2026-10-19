#!/usr/bin/env python3
"""Benchmark one explicit model variant on a Rockchip board, without publishing."""

import argparse
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

DEFAULT_PROMPT = (
    "Write a detailed explanation of how a computer works, covering the processor, "
    "memory, storage, operating system, and networking. Use complete paragraphs "
    "and continue for at least 500 words."
)
APP = Path(__file__).resolve().parents[1] / "app"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--platform", choices=["rk3576", "rk3588"], required=True)
    parser.add_argument("--quantization", choices=["w4a16_g128", "w8a8"], required=True)
    parser.add_argument("--date", required=True, help="Measurement date YYYY-MM-DD from the PC; board clocks may be wrong")
    parser.add_argument("--output", type=Path, required=True, help="New report directory, outside the source repository")
    parser.add_argument("--vision", type=Path)
    parser.add_argument("--vision-sha256")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--audio-sha256")
    parser.add_argument("--embed-flash", type=int, choices=[0, 1], default=0)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--baseline-seconds", type=float, default=5)
    parser.add_argument("--stable-seconds", type=float, default=10)
    parser.add_argument("--sample-seconds", type=float, default=0.05)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--docker-image", help="Run in this explicit RKLLM 1.3.1 environment image on the board")
    parser.add_argument("--image-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if datetime.date.fromisoformat(args.date).isoformat() != args.date:
            raise ValueError("non-canonical date")
    except ValueError:
        parser.error("--date must be YYYY-MM-DD")
    if min(args.context, args.runs, args.warmup_tokens) < 1 or args.tokens < 2:
        parser.error("context/runs/warmup-tokens must be positive; tokens must be at least 2")
    if args.tokens >= args.context or args.warmup_tokens >= args.context:
        parser.error("generation lengths must be smaller than context")
    durations = [args.baseline_seconds, args.stable_seconds, args.sample_seconds]
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        parser.error("sampling durations must be positive")
    if args.platform == "rk3588" and args.quantization != "w8a8":
        parser.error("this workflow supports RK3588 W8A8")
    for key in ["model", "vision", "audio"]:
        path = getattr(args, key)
        expected = getattr(args, key + "_sha256")
        if path and (not path.is_file() or not expected or len(expected) != 64
                     or any(c not in "0123456789abcdefABCDEF" for c in expected)):
            parser.error(f"--{key} needs an existing file and its 64-digit SHA-256")
        if expected and not path:
            parser.error(f"--{key}-sha256 needs --{key}")
    args.output = args.output.resolve()
    if (args.output / "result.json").exists():
        parser.error("report already exists; choose a new --output directory")
    return args


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hash_stream(source)


def hash_stream(source):
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def docker_run(args):
    if subprocess.check_output(["docker", "ps", "-q"], text=True).strip():
        raise RuntimeError("The board has running containers; use an idle board for comparable measurements")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.docker_image], text=True
    ).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "--rm", "--name", f"rkllm-benchmark-{os.getpid()}",
               "--privileged", "--network", "none", "--ulimit", "core=0",
               "-v", f"{APP.parent}:/benchmark/source:ro",
               "-v", f"{args.output}:/benchmark/output"]
    forwarded = []
    for key, value in vars(args).items():
        if value is None or key in {"docker_image", "image_id"}:
            continue
        if key in {"model", "vision", "audio"}:
            path = value.resolve()
            value = f"/benchmark/{key}/{path.name}"
            command += ["-v", f"{path}:{value}:ro"]
        elif key == "output":
            value = "/benchmark/output"
        forwarded += ["--" + key.replace("_", "-"), str(value)]
    command += ["--entrypoint", "python3", args.docker_image,
                "/benchmark/source/scripts/benchmark_model.py", *forwarded,
                "--image-id", image_id]
    name = command[command.index("--name") + 1]
    try:
        with (args.output / "runtime.log").open("w") as log:
            return_code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    except KeyboardInterrupt:
        return_code = 130
    finally:
        subprocess.run(["docker", "stop", "--time", "30", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not (args.output / "result.json").exists():
        save(args.output / "result.json", {"success": False, "error": "Native process exited without a report",
                                          "return_code": return_code, "image_id": image_id})
    result = json.loads((args.output / "result.json").read_text())
    print(json.dumps(result.get("summary", {"success": result.get("success"), "error": result.get("error")})))
    return 0 if return_code == 0 and result.get("success") else 1


class MemorySampler:
    def __init__(self, interval):
        self.interval = interval
        self.phase = "baseline"
        self.rows = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                mem = {line.split(":")[0]: int(line.split()[1]) * 1024
                       for line in Path("/proc/meminfo").read_text().splitlines()}
                process = {line.split(":")[0]: int(line.split()[1]) * 1024
                           for line in Path("/proc/self/status").read_text().splitlines()
                           if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:"))}
                self.rows.append({"time": time.perf_counter(), "phase": self.phase,
                                  "board_used": mem["MemTotal"] - mem["MemAvailable"],
                                  "swap_used": mem["SwapTotal"] - mem["SwapFree"], **process})
            except Exception as error:
                self.errors.append(str(error))
                return
            self.stop.wait(self.interval)

    def finish(self):
        self.stop.set()
        self.thread.join()
        return {phase: {key: {"max": max(row[key] for row in rows),
                             "median": statistics.median(row[key] for row in rows)}
                        for key in ["board_used", "swap_used", "VmRSS", "VmHWM", "VmSwap"]}
                for phase in ["baseline", "startup", "ready", "warmup", "generation"]
                if (rows := [row for row in self.rows if row["phase"] == phase])}


def clocks():
    values = {}
    for pattern in ["/sys/devices/system/cpu/cpufreq/policy*/scaling_*freq",
                    "/sys/devices/system/cpu/cpufreq/policy*/scaling_governor",
                    "/sys/class/devfreq/*/cur_freq", "/sys/class/devfreq/*/governor",
                    "/sys/class/thermal/thermal_zone*/temp"]:
        import glob
        for path in glob.glob(pattern):
            try:
                values[path] = Path(path).read_text().strip()
            except OSError:
                pass
    return values


def native_run(args):
    sys.path.insert(0, str(APP))
    import rkllm_types as rt
    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"success": False, "measurement_date": args.date, "image_id": args.image_id,
              "platform": args.platform, "quantization": args.quantization,
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "sampling": {"top_k": 1, "top_p": 1.0, "temperature": 0.0,
                           "repeat_penalty": 1.1, "thinking": False, "ignore_eos": True},
              "artifacts": {}, "runs": [], "clocks_before": clocks(),
              "ttft_definition": "Milliseconds from immediately before rkllm_run to the first native token callback; excludes HTTP transport and model loading"}
    handle = ctypes.c_void_p()
    encoders = []
    sampler = MemorySampler(args.sample_seconds)
    lock = open("/tmp/rkllm-benchmark.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in ["model", "vision", "audio"]:
            path = getattr(args, key)
            if path:
                actual = sha256(path)
                if actual.lower() != getattr(args, key + "_sha256").lower():
                    raise ValueError(f"{key} SHA-256 mismatch")
                report["artifacts"][key] = {"file": path.name, "sha256": actual, "bytes": path.stat().st_size}
        report["runtime_sha256"] = sha256(Path("/usr/lib/librkllmrt.so"))
        manifest = json.loads((APP.parent / "runtime" / "rkllm-manifest.json").read_text())
        if report["runtime_sha256"] != manifest["sha256"]["lib/librkllmrt.so"]:
            raise RuntimeError("Native library does not match this checkout's RKLLM runtime manifest")
        report["runtime_version"] = manifest["version"]
        report["script_sha256"] = sha256(Path(__file__))
        report["bindings_sha256"] = sha256(APP / "rkllm_types.py")
        if args.vision or args.audio:
            report["rknn_runtime_sha256"] = sha256(Path("/usr/lib/librknnrt.so"))
        sampler.thread.start()
        time.sleep(args.baseline_seconds)
        sampler.phase = "startup"
        lib = ctypes.CDLL("/usr/lib/librkllmrt.so", mode=ctypes.RTLD_GLOBAL)
        lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(rt.RKLLMParam), ctypes.POINTER(rt.RKLLMCallback)]
        lib.rkllm_init.restype = ctypes.c_int
        lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(rt.RKLLMInput), ctypes.POINTER(rt.RKLLMInferParam), ctypes.c_void_p]
        lib.rkllm_run.restype = ctypes.c_int
        lib.rkllm_clear_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        lib.rkllm_clear_kv_cache.restype = ctypes.c_int
        current = {}

        @rt.callback_type
        def callback(result, userdata, state):
            try:
                if state == 3:
                    current["errors"].append("RKLLM_RUN_ERROR")
                if result:
                    value = result.contents
                    elapsed = (time.perf_counter() - current.get("start", time.perf_counter())) * 1000
                    if state in (0, 1) and value.token_id >= 0:
                        current.setdefault("ttft_ms", elapsed)
                    if state == 0 and value.text:
                        current.setdefault("first_text_ms", elapsed)
                        current["text"].append(value.text.decode("utf-8", errors="replace"))
                    if state == 2:
                        current["perf"] = {key: getattr(value.perf, key) for key, _ in rt.RKLLMPerfStat._fields_}
            except Exception as error:
                current.setdefault("errors", []).append(str(error))
            return 0

        callbacks = rt.RKLLMCallback()
        callbacks.result_callback = callback
        params = rt.RKLLMParam()
        params.model_path = os.fsencode(args.model)
        params.max_context_len = args.context
        params.max_new_tokens = args.tokens
        params.top_k, params.top_p, params.temperature = 1, 1.0, 0.0
        params.repeat_penalty = 1.1
        params.mirostat_tau, params.mirostat_eta = 5.0, 0.1
        params.ignore_eos_token = True
        params.extend_param.base_domain_id = 1
        params.extend_param.embed_flash = args.embed_flash
        params.extend_param.n_batch = 1
        params.extend_param.enabled_cpus_mask = 0xF0
        params.extend_param.enabled_cpus_num = 4
        start = time.perf_counter()
        status = lib.rkllm_init(ctypes.byref(handle), ctypes.byref(params), ctypes.byref(callbacks))
        if status:
            raise RuntimeError(f"rkllm_init returned {status}")
        if args.vision or args.audio:
            from fastapi_server_vlm import RKNNImageEncoder, RKNNAudioEncoder
            if args.vision:
                encoders.append(RKNNImageEncoder(str(args.vision), 2 if args.platform == "rk3576" else 3))
            if args.audio:
                encoders.append(RKNNAudioEncoder(str(args.audio), 1))
        report["load_seconds"] = time.perf_counter() - start
        sampler.phase = "ready"
        time.sleep(args.stable_seconds)
        for index in range(args.runs + 1):
            sampler.phase = "warmup" if index == 0 else "generation"
            if lib.rkllm_clear_kv_cache(handle, 0, None, None):
                raise RuntimeError("KV cache reset failed")
            current.clear()
            current.update(text=[], errors=[])
            prompt = args.prompt.encode()
            value = rt.RKLLMInput()
            value.role, value.enable_thinking, value.input_type = b"user", False, 0
            value.input_data.prompt_input = prompt
            infer = rt.RKLLMInferParam()
            infer.mode, infer.keep_history = 0, 0
            infer.max_new_tokens = args.warmup_tokens if index == 0 else args.tokens
            current["start"] = time.perf_counter()
            status = lib.rkllm_run(handle, ctypes.byref(value), ctypes.byref(infer), None)
            perf = current.get("perf", {})
            milliseconds, tokens = perf.get("generate_time_ms", 0), perf.get("generate_tokens", 0)
            row = {"warmup": index == 0, "return_code": status,
                   "wall_ms": (time.perf_counter() - current["start"]) * 1000,
                   "ttft_ms": current.get("ttft_ms"), "first_text_ms": current.get("first_text_ms"),
                   "perf": perf, "text": "".join(current["text"]), "errors": current["errors"]}
            if milliseconds > 0 and tokens > 0:
                row["tok_s"] = 1000 * tokens / milliseconds
            report["runs"].append(row)
            save(args.output / "partial.json", report)
            if status or row["errors"] or not row.get("tok_s") or row["ttft_ms"] is None:
                raise RuntimeError("Missing/failed native generation, timing, or token callback")
        measured = report["runs"][1:]
        report["summary"] = {"median_tok_s": statistics.median(row["tok_s"] for row in measured),
                             "median_ttft_ms": statistics.median(row["ttft_ms"] for row in measured)}
        report["success"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if sampler.thread.ident is not None:
            report["memory"] = sampler.finish()
        report["clocks_after"] = clocks()
        for encoder in reversed(encoders):
            try:
                encoder.close()
            except Exception as error:
                report["success"] = False
                report["cleanup_error"] = str(error)
        if handle.value:
            lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
            lib.rkllm_destroy.restype = ctypes.c_int
            if lib.rkllm_destroy(handle):
                report["success"] = False
                report["cleanup_error"] = "rkllm_destroy failed"
        lock.close()
    if sampler.errors:
        report["success"], report["memory_errors"] = False, sampler.errors
    if report["success"] and not all(phase in report.get("memory", {}) for phase in ["baseline", "startup", "ready", "generation"]):
        report["success"], report["error"] = False, "Missing memory samples for a measurement phase"
    if report["success"]:
        mem = report["memory"]
        baseline = mem["baseline"]["board_used"]["median"]
        summary = report["summary"]
        summary["startup_peak_delta_gb"] = (mem["startup"]["board_used"]["max"] - baseline) / 1e9
        summary["stable_delta_gb"] = (mem["ready"]["board_used"]["median"] - baseline) / 1e9
        summary["swap_growth_gb"] = (max(row["swap_used"] for row in sampler.rows) - mem["baseline"]["swap_used"]["median"]) / 1e9
        report["notion_ready"] = summary["swap_growth_gb"] <= 0.01 and min(summary["startup_peak_delta_gb"], summary["stable_delta_gb"]) >= 0
        if report["notion_ready"]:
            base = f"{args.platform}_{args.quantization}"
            # TTFT deliberately stays in the report, never in the Notion payload.
            properties = {base + "_tok_s": round(summary["median_tok_s"], 2),
                          base + "_startup_GB": round(summary["startup_peak_delta_gb"], 2),
                          base + "_stable_GB": round(summary["stable_delta_gb"], 2),
                          "date:Benchmark_Checked:start": args.date,
                          "date:Benchmark_Checked:is_datetime": 0}
            save(args.output / "notion-properties.json", properties)
    save(args.output / "memory-samples.json", sampler.rows)
    save(args.output / "result.json", report)
    print(json.dumps(report.get("summary", {"success": False, "error": report.get("error")})))
    return 0 if report["success"] else 1


def main():
    args = parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Run on the target Linux Rockchip board")
    if args.docker_image:
        return docker_run(args)
    return native_run(args)


if __name__ == "__main__":
    sys.exit(main())
