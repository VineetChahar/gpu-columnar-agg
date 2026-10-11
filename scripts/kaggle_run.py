#!/usr/bin/env python3
"""Pack the repo, push it as a Kaggle GPU script kernel, poll, pull outputs into runs/<UTC>/.

usage: kaggle_run.py [--cmd "make verify"] [--accel NvidiaTeslaT4Highmem] [--timeout 32400]
Exits non-zero if the kernel fails (and prints the last 100 log lines).
"""
import argparse, base64, datetime, io, json, os, re, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = str(ROOT / ".venv/bin/kaggle") if (ROOT / ".venv/bin/kaggle").exists() else "kaggle"
INCLUDE = ["src", "bench", "tests", "scripts", "CMakeLists.txt", "Makefile"]

DRIVER = r'''
import base64, io, os, subprocess, sys, tarfile, shutil, time
PAYLOAD = "__PAYLOAD__"
CMD = __CMD__
W = "/kaggle/working"
os.makedirs(W, exist_ok=True)
repo = "/kaggle/repo"
shutil.rmtree(repo, ignore_errors=True); os.makedirs(repo)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz").extractall(repo)
env = dict(os.environ, RESULTS_DIR=W)
log = open(os.path.join(W, "run.log"), "w")
def run(c):
    print("$ " + c, flush=True); log.write("$ " + c + "\n")
    p = subprocess.Popen(c, shell=True, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line); log.write(line)
    p.wait(); log.flush(); return p.returncode
run("nvidia-smi; nvcc --version; nproc; free -g")
env["PATH"] = "/usr/local/cuda/bin:" + env["PATH"]
rc = run(CMD)
import json
jl = os.path.join(W, "results.jsonl")
if os.path.exists(jl):
    rows = [json.loads(l) for l in open(jl) if l.strip()]
    json.dump(rows, open(os.path.join(W, "results.json"), "w"))
log.write("EXIT_CODE=%d\n" % rc); log.close()
print("EXIT_CODE=%d" % rc)
sys.exit(rc)
'''

def pack():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in INCLUDE:
            p = ROOT / name
            if p.exists():
                tf.add(p, arcname=name, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()

def sh(args, **kw):
    return subprocess.run([KAGGLE] + args, capture_output=True, text=True, **kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="make bench")
    ap.add_argument("--accel", default="NvidiaTeslaT4Highmem")
    ap.add_argument("--timeout", type=int, default=32400)
    ap.add_argument("--slug", default="gpu-columnar-agg")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--max-wait", type=int, default=6 * 3600)
    ap.add_argument("--cpu", action="store_true", help="no accelerator (compile checks; does not use GPU quota)")
    a = ap.parse_args()

    user = re.search(r"username: (\S+)", sh(["config", "view"]).stdout).group(1)
    ref = f"{user}/{a.slug}"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "runs" / stamp
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "main.py").write_text(DRIVER.replace("__PAYLOAD__", pack()).replace("__CMD__", repr(a.cmd)))
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": ref, "title": a.slug, "code_file": "main.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": not a.cpu,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}))
        # The account allows 2 concurrent batch GPU sessions (shared with other projects): wait for a slot.
        waited = 0
        while True:
            args = ["kernels", "push", "-p", str(td), "-t", str(a.timeout)] + ([] if a.cpu else ["--accelerator", a.accel])
            r = sh(args)
            txt = r.stdout + r.stderr
            if "session count" in txt and waited < a.max_wait:
                print(f"[queue] GPU sessions busy, retrying in 120 s (waited {waited} s)", flush=True)
                time.sleep(120); waited += 120
                continue
            break
        print(r.stdout, r.stderr)
        if r.returncode != 0 or "error" in txt.lower():
            sys.exit("push failed")
    (out / "meta.json").write_text(json.dumps({"ref": ref, "cmd": a.cmd, "accel": a.accel, "utc": stamp}))

    t0 = time.time(); status = ""
    while True:
        time.sleep(a.poll)
        s = sh(["kernels", "status", ref]); txt = (s.stdout + s.stderr).strip()
        m = re.search(r'status "?([\w.]+)"?', txt); status = m.group(1) if m else txt
        print(f"[{int(time.time()-t0):5d}s] {status}", flush=True)
        if any(k in status.upper() for k in ("COMPLETE", "ERROR", "CANCEL")):
            break

    r = sh(["kernels", "output", ref, "-p", str(out), "-o"]); print(r.stdout[-500:], r.stderr[-500:])
    logs = sorted(out.glob("*.log"), key=lambda p: p.name != "run.log")  # our plain log first
    for f in ("results.json",):
        if (out / f).exists(): print("pulled", f)
    ok = "COMPLETE" in status.upper() and "EXIT_CODE=0" in "".join(p.read_text(errors="ignore")[-2000:] for p in logs)
    if not ok:
        print(f"KERNEL FAILED ({status}); last 100 log lines:")
        for p in logs[:1]:
            print("".join(p.read_text(errors="ignore").splitlines(True)[-100:]))
        sys.exit(1)
    print("OK ->", out)

if __name__ == "__main__":
    main()
