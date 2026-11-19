"""
load_check.py
=============
Verified model loading for an unreliable disk.

After a model is loaded, five fixed reference questions are scored and their
predicted options and confidences compared with values from earlier, validated
loads of the same file (fingerprints.json). On a mismatch the model is freed,
the file's cached pages are dropped (so the next read comes from the disk again,
not from a corrupted cache) and the load is repeated, up to three times.
"""
import gc
import json
import os
import time

from energy_per_item import run_loglik


def drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (OSError, AttributeError):
        pass


def load_verified(make_llm, model_path, fp_file, fp_key, items, tries=3):
    """Returns (llm, info). Raises SystemExit if no load reproduces the fingerprint."""
    ref = json.load(open(fp_file))[fp_key] if fp_file else None
    by_key = {(it["task"], str(it["doc_id"])): it for it in items}
    info = {"fingerprint_key": fp_key, "attempts": []}
    for attempt in range(1, tries + 1):
        drop_cache(model_path)
        t0 = time.perf_counter()
        llm = make_llm()
        load_s = time.perf_counter() - t0
        if ref is None:
            info.update(load_seconds=load_s, verified=None)
            return llm, info
        bad = []
        for task, doc, pred, conf in ref:
            p, c, *_ = run_loglik(llm, by_key[(task, str(doc))], False)
            if p != pred or abs(c - conf) > 2e-6:
                bad.append((task, doc, pred, conf, p, round(c, 6)))
        info["attempts"].append({"attempt": attempt, "load_seconds": round(load_s, 1), "mismatches": len(bad)})
        if not bad:
            print(f"[load check] {fp_key}: fingerprint reproduced on attempt {attempt}")
            info.update(load_seconds=load_s, verified=True)
            return llm, info
        print(f"[load check] {fp_key}: attempt {attempt} FAILED ({len(bad)}/{len(ref)} mismatches), reloading")
        del llm
        gc.collect()
        time.sleep(5)
    raise SystemExit(f"[load check] {fp_key}: no load reproduced the fingerprint in {tries} attempts; setting skipped")
