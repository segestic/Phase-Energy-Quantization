"""
verify_gguf_sha.py
==================
Compare the SHA-256 of local GGUF files with the hashes Hugging Face publishes
for segestic/MediLITE-QA-GGUF. Standard library only; needs internet.

Usage:
    python3 verify_gguf_sha.py MediLITE-QA.Q4_K_M.gguf MediLITE-QA.Q8_0.gguf
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

REPO = "segestic/MediLITE-QA-GGUF"

req = urllib.request.Request(f"https://huggingface.co/api/models/{REPO}/tree/main",
                             headers={"User-Agent": "verify-gguf/1.0"})
tree = json.load(urllib.request.urlopen(req, timeout=60))
hf = {e["path"]: ((e.get("lfs") or {}).get("oid"), e.get("size")) for e in tree}

lines = []
for path in sys.argv[1:]:
    name = os.path.basename(path)
    want, want_size = hf.get(name, (None, None))
    size = os.path.getsize(path)
    h, done, t0 = hashlib.sha256(), 0, time.time()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(16 << 20)
                if not chunk:
                    break
                h.update(chunk)
                done += len(chunk)
                print(f"\r  {name}: {done / size:6.1%}", end="", flush=True)
        got = h.hexdigest()
        verdict = "MATCH" if got == want else "DIFFERENT"
    except OSError as e:
        got, verdict = f"read failed at {done / 2**30:.2f} GiB ({e})", "UNREADABLE"
    print()
    lines += [name,
              f"  Hugging Face  {want}  ({want_size} bytes)",
              f"  local         {got}  ({size} bytes)",
              f"  => {verdict}   ({time.time() - t0:.0f} s)"]

print("\n".join(lines))
with open("verify_gguf_sha.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\nwritten to verify_gguf_sha.txt")
