"""
Instruction-tuning data tests. Run:  python tests/test_instruct.py <tokenizer.json>

Checks that the loss sees ONLY the response + <|endoftext|>, that padding is
ignored, and that training tokens match what the model sees at inference.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.instruct_format import build_prompt, encode_example, encode_prompt  # noqa: E402
from train.instruct import EOT_ID, PAD_ID, collate  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


def run(tok):
    assert tok.token_to_id("<|endoftext|>") == EOT_ID and tok.token_to_id("<|pad|>") == PAD_ID

    p = build_prompt("Name a colour.", "")
    assert p == "<|user|>\nName a colour.\n<|assistant|>\n", repr(p)
    p = build_prompt("Summarise.", "Cats sleep a lot.")
    assert p == "<|user|>\nSummarise.\n<|context|>\nCats sleep a lot.\n<|assistant|>\n", repr(p)
    print("ok  prompt template (with and without context)")

    ids, plen = encode_example(tok, "What is the code for Room 12?", "The code for Room 12 is 4821.",
                               "The code is 4821.")
    assert ids[:plen] == encode_prompt(tok, "What is the code for Room 12?", "The code for Room 12 is 4821.")
    print("ok  training prompt tokens == inference prompt tokens")

    ids2, plen2 = encode_example(tok, "Say hi.", "", "Hi!")
    x, y = collate([ids, ids2], [plen, plen2], "cpu")
    for row, (e, p) in enumerate([(ids, plen), (ids2, plen2)]):
        graded = [int(t) for t in y[row] if t != -100]
        assert graded == e[p:], (graded, e[p:])
        assert tok.decode(graded, skip_special_tokens=False).endswith("<|endoftext|>")
        # x[t] predicts y[t]: the first graded target is predicted from the <|assistant|>\n position
        first = (y[row] != -100).nonzero()[0].item()
        assert first == p - 1, (first, p)
    assert (x[1, len(ids2) - 1:] == PAD_ID).all() and (y[1, len(ids2) - 1:] == -100).all()
    print(f"ok  loss mask: only response + <|endoftext|> graded ({len(ids) - plen} of {len(ids) - 1} targets); "
          "padding ignored")
    print(f"    graded text: {tok.decode(ids[plen:], skip_special_tokens=False)!r}")
    print("\nall instruction-data tests passed")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/tokenizer/tokenizer.json"
    run(Tokenizer.from_file(path))
