"""
End-to-end test of the long-context data pipeline on tiny fake data (CPU, no internet):
prepare_pg19 -> loader rows -> pretrain (A, then C v2 from A) -> far_recall_loss,
prepare_babilong -> babilong_eval.
Run:  python tests/test_data_v2.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import SPECIAL_TOKENS  # noqa: E402
from data.loader import TokenLoader  # noqa: E402
from data.prepare_pretrain import file_sha  # noqa: E402
from eval.far_recall_loss import buckets_for  # noqa: E402

WORDS = ("the cat dog house garden kitchen office hallway bedroom bathroom john mary went to "
         "journeyed moved travelled back where is question story read answer").split()


def run(*cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{cmd[0]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}"
    return r.stdout


def make_tokenizer(path, rng):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    text = [" ".join(rng.choice(WORDS, 200)) + " Question: Answer with one word. 0123456789" for _ in range(50)]
    tok.train_from_iterator(text, trainers.BpeTrainer(vocab_size=300, special_tokens=SPECIAL_TOKENS,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(path)
    return Tokenizer.from_file(path)


def test_buckets():
    b = buckets_for(np.array([5, 6, 5, 7, 6, 5]), window=2)
    # i=0 new, 1 new, 2: 5 last at 0 (2 back) near, 3 new, 4: 6 last at 1 (3 back) far, 5: 5 last at 2 (3 back) far
    assert b.tolist() == [2, 2, 0, 2, 1, 1], b
    print("ok  far-recall buckets: near <= window < far, never-seen = new")


def main():
    test_buckets()
    rng = np.random.default_rng(0)
    d = tempfile.mkdtemp(prefix="afml_data_v2_")
    tok_path = os.path.join(d, "tokenizer.json")
    tok = make_tokenizer(tok_path, rng)
    sha = file_sha(tok_path)

    # fake FineWeb (existing pretrain data format)
    fw = os.path.join(d, "fw"); os.makedirs(fw)
    rng.integers(5, tok.get_vocab_size(), 50_000).astype(np.uint16).tofile(os.path.join(fw, "train.bin"))
    json.dump({"tokens": {"train": 50_000, "val": 0}, "tokenizer_sha": sha, "source": "fake"},
              open(os.path.join(fw, "meta.json"), "w"))
    # fake PG-19 books
    books = os.path.join(d, "books")
    for split, n in (("train", 24), ("validation", 2), ("test", 3)):
        os.makedirs(os.path.join(books, split))
        for i in range(n):
            open(os.path.join(books, split, f"{split}_{i:02d}.txt"), "w").write(
                " ".join(rng.choice(WORDS, int(rng.integers(300, 1500)))))

    # ---- PG-19 mix ------------------------------------------------------------
    out = os.path.join(d, "pg19mix")
    run("data/prepare_pg19.py", "--tokenizer", tok_path, "--fineweb_dir", fw, "--out_dir", out,
        "--local_books", books, "--seq_len", "64", "--train_tokens", str(130 * 65), "--max_rows_per_book", "8")
    meta = json.load(open(os.path.join(out, "meta.json")))
    R = meta["row_len"]
    assert meta["format"] == "rows" and R == 65 and meta["tokenizer_sha"] == sha
    train = np.fromfile(os.path.join(out, "train.bin"), dtype=np.uint16).reshape(-1, R)
    src = np.load(os.path.join(out, "train_source.npy"))
    assert len(train) == len(src) == meta["rows"]["train"] == 130
    assert abs(src.mean() - 0.25) < 0.02, src.mean()
    # every PG-19 row is a contiguous slice of one tokenized training book
    book_ids = [np.array(tok.encode(open(os.path.join(books, "train", f)).read()).ids + [0], np.uint16)
                for f in sorted(os.listdir(os.path.join(books, "train")))]
    for row in train[src == 0][:20]:
        assert any(any(np.array_equal(row, b[k * R:(k + 1) * R]) for k in range(len(b) // R)) for b in book_ids)
    assert (train[src == 0].reshape(-1)[:1] >= 0).all()
    offs = np.load(os.path.join(out, "test_offsets.npy"))
    assert len(offs) == 4 and offs[-1] == meta["test_tokens"]
    print(f"ok  PG-19 mix: {len(train)} rows of {R} (25% FineWeb), each PG-19 row = one continuous book slice; "
          f"val {meta['rows']['val']} rows; {meta['test_books']} whole test books")

    # ---- loader in row mode -----------------------------------------------------
    L = TokenLoader(out, seq_len=64, batch_size=8)
    x, y = L.get_batch("train")
    rows = {tuple(r) for r in train.tolist()}
    for xi, yi in zip(x.numpy(), y.numpy()):
        assert tuple(np.concatenate([xi, yi[-1:]]).tolist()) in rows, "batch row is not a whole stored row"
    assert torch.equal(x[:, 1:], y[:, :-1])
    nval = sum(1 for _ in L.val_batches())
    try:
        TokenLoader(out, seq_len=32, batch_size=8)
        raise AssertionError("wrong seq_len accepted")
    except AssertionError as e:
        assert "use --seq_len 64" in str(e), e
    print(f"ok  loader: row data -> whole rows only, x/y shifted; {nval} val batches; wrong --seq_len refused")

    # ---- pretrain: A, then C v2 started from A's backbone -------------------------
    common = ["--tiny", "--data_dir", out, "--seq_len", "64", "--max_steps", "4", "--batch_size", "4",
              "--grad_accum", "1", "--eval_every", "4", "--eval_batches", "1", "--warmup_steps", "1",
              "--device", "cpu", "--resume", "none"]
    run("train/pretrain.py", *common, "--variant", "A", "--out_dir", os.path.join(d, "A"))
    a_ck = os.path.join(d, "A", "latest.pt")
    o = run("train/pretrain.py", *common, "--variant", "C", "--memory_version", "2", "--init_from", a_ck,
            "--out_dir", os.path.join(d, "C"))
    assert "backbone initialised" in o
    a_final = [f for f in os.listdir(os.path.join(d, "A")) if f.startswith("variant-A_pretrain")][0]
    c_final = [f for f in os.listdir(os.path.join(d, "C")) if f.startswith("variant-C_pretrain")][0]
    A, C = os.path.join(d, "A", a_final), os.path.join(d, "C", c_final)
    print("ok  pretrain on rows: A, and C v2 initialised from A's backbone")

    # ---- far-recall loss --------------------------------------------------------
    o = run("eval/far_recall_loss.py", "--data_dir", out, "--ckpt", f"A={A}", f"C={C}", "--seg_len", "64",
            "--window", "16", "--device", "cpu", "--out", os.path.join(d, "far.json"))
    res = json.load(open(os.path.join(d, "far.json")))
    assert set(res) == {"A", "C"} and res["A"]["far"]["tokens"] > 0
    assert res["A"]["far"]["tokens"] == res["C"]["far"]["tokens"], "variants scored on different tokens"
    print(f"ok  far-recall loss runs with memory carried across segments ({res['A']['far']['tokens']} far tokens)")

    # ---- BABILong ----------------------------------------------------------------
    bl = os.path.join(d, "bl")
    places = ["garden", "kitchen", "office", "hallway", "bedroom", "bathroom"]

    def fake_rows(n, seed, filler):
        r = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            who, where = r.choice(["john", "mary"]), r.choice(places)
            story = " ".join(r.choice(WORDS, filler)) + f" {who} went to the {where}. " + " ".join(r.choice(WORDS, filler))
            out.append({"input": story, "question": f"where is {who}?", "target": where})
        return out

    for repo, seed in (("babilong-train-5k-samples", 1), ("babilong-1k-samples", 2)):
        for length, filler in (("0k", 5), ("1k", 60), ("2k", 120), ("4k", 400)):
            for task in ("qa1", "qa2"):
                p = os.path.join(bl, repo, length); os.makedirs(p, exist_ok=True)
                rows_ = fake_rows(40, seed * 100 + filler + len(task), filler)
                if repo == "babilong-1k-samples" and length == "1k" and task == "qa1":
                    tr = json.loads(open(os.path.join(bl, "babilong-train-5k-samples", "1k", "qa1.jsonl")).readline())
                    rows_[0] = tr                                   # a story that is also in train
                with open(os.path.join(p, f"{task}.jsonl"), "w") as f:
                    f.writelines(json.dumps(r_) + "\n" for r_ in rows_)
    dolly = os.path.join(d, "dolly"); os.makedirs(dolly)
    from data.instruct_format import encode_example
    for split, n in (("train", 30), ("val", 5)):
        dd = {"ids": [], "prompt_len": [], "raw": []}
        for i in range(n):
            ids, p = encode_example(tok, f"say {WORDS[i % 10]}", "", WORDS[i % 7])
            dd["ids"].append(ids); dd["prompt_len"].append(p)
            dd["raw"].append({"instruction": f"say {WORDS[i % 10]}", "context": "", "response": WORDS[i % 7],
                              "category": "dolly"})
        torch.save(dd, os.path.join(dolly, f"{split}.pt"))
    json.dump({"tokenizer_sha": sha}, open(os.path.join(dolly, "meta.json"), "w"))
    ib = os.path.join(d, "ib")
    # keep only qa1 in the first train pick so the planted duplicate is guaranteed to be in train
    run("data/prepare_babilong.py", "--tokenizer", tok_path, "--dolly_dir", dolly, "--out_dir", ib,
        "--local_dir", bl, "--tasks", "qa1", "qa2", "--train_per_cell", "40", "--val_per_cell", "0",
        "--test_per_cell", "40", "--max_len", "1200", "--test_lengths", "1k", "2k")
    m = json.load(open(os.path.join(ib, "meta.json")))
    tr = torch.load(os.path.join(ib, "train.pt")); va = torch.load(os.path.join(ib, "val.pt"))
    te = torch.load(os.path.join(ib, "test.pt"))
    assert m["train_examples"]["dolly"] == 30 and m["train_examples"]["babilong"] > 0
    assert m["cells"]["qa1/4k"]["train"]["dropped_too_long"] > 0, "max_len not applied"
    assert va["raw"][0]["category"] == "dolly", "Dolly val must come first"
    assert m["test_cells"]["qa1/1k"]["removed_seen_in_train"] >= 1, "train/test overlap not removed"
    assert te["choices"]["qa1"] == sorted(places)
    ex = next(r for r in tr["raw"] if r["category"].startswith("babilong"))
    assert "\n\nQuestion: where is" in ex["context"] and ex["context"].rstrip().endswith("?")
    i = next(k for k, r in enumerate(tr["raw"]) if r["category"].startswith("babilong"))
    assert tok.decode(tr["ids"][i][tr["prompt_len"][i]:], skip_special_tokens=True).strip() == ex["response"]
    print(f"ok  BABILong: {m['train_examples']['babilong']} train examples mixed into {m['train_examples']['dolly']} "
          f"Dolly; question at the end of the story; too-long dropped; train/test overlap removed")

    o = run("eval/babilong_eval.py", "--data_dir", ib, "--tokenizer", tok_path, "--ckpt", f"A={A}", f"C={C}",
            "--device", "cpu", "--max_per_cell", "3", "--out", os.path.join(d, "bl.json"))
    br = json.load(open(os.path.join(d, "bl.json")))
    assert set(br["A"]) == {"qa1/1k", "qa1/2k", "qa2/1k", "qa2/2k"}
    print("ok  BABILong eval scores every answer choice over the full prompt, per task and length")
    print("\nall data v2 tests passed")


if __name__ == "__main__":
    main()
