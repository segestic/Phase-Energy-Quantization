"""
check_gguf.py
=============
Sanity check for a GGUF file before measuring with it: prints the file type and
architecture, the spread of the next-token logits and the five most likely next
tokens for a fixed prompt. A working model has a clearly non-zero logit standard
deviation and a sensible top token (e.g. " Paris"); a broken file shows std 0.

Usage:  python check_gguf.py MODEL.gguf [MODEL2.gguf ...]
"""
import sys
import numpy as np
from llama_cpp import Llama

for path in sys.argv[1:]:
    m = Llama(model_path=path, n_ctx=512, n_threads=4, verbose=False)
    m.eval(m.tokenize(b"Question: What is the capital of France?\nAnswer:", add_bos=False))
    lg = np.ctypeslib.as_array(m._ctx.get_logits_ith(-1), shape=(m.n_vocab(),)).copy()
    top = [m.detokenize([int(i)]) for i in np.argsort(lg)[-5:][::-1]]
    print(path.split("/")[-1], "file_type", m.metadata.get("general.file_type"),
          "arch", m.metadata.get("general.architecture"),
          "| nan", int(np.isnan(lg).sum()), "std", round(float(lg.std()), 3), "top", top)
