"""Command line: python -m scratchllm <command>."""

import argparse
import math
import sys
from pathlib import Path

from . import evals
from .chat_format import chat_reply, chat_transform, prepare_chat
from .config import load_config
from .data import prepare, read_text_docs
from .registry import RegistryError, load_registry, render_licences, verify
from .sample import generate
from .tokenizer import Tokenizer
from .train import TrainingDiverged, load_model, pick_device, train

_EVAL_BATCHES, _EVAL_BATCH_SIZE, _EVAL_SEED = 50, 16, 1234


def _first_megabytes(docs, megabytes: float):
    budget = megabytes * 1_000_000
    for doc in docs:
        if budget <= 0:
            return
        budget -= len(doc)
        yield doc


def _load(args):
    """The model and tokenizer of a command, checked to belong together."""
    model, tok = load_model(args.ckpt, pick_device()), Tokenizer.load(args.tokenizer)
    if model.cfg.vocab_size != tok.vocab_size:
        raise ValueError(
            f"{args.tokenizer} has a vocabulary of {tok.vocab_size} but {args.ckpt} "
            f"was trained with a vocabulary of {model.cfg.vocab_size}"
        )
    return model, tok


def _verify(args) -> int:
    datasets = [ds for ds in load_registry(args.registry) if args.id in (None, ds.id)]
    if not datasets:
        raise RegistryError(f"no dataset {args.id!r} in {args.registry}" if args.id else f"{args.registry} is empty")
    for ds in datasets:
        verify(ds, args.raw)
        print(f"ok  {ds.id} ({len(ds.files)} files)")
    return 0


def _licences(args) -> int:
    Path(args.output).write_text(render_licences(load_registry(args.registry)), encoding="utf-8")
    return 0


def _tokenizer_train(args) -> int:
    docs = _first_megabytes(read_text_docs(args.input, args.separator), args.sample_mb)
    tok = Tokenizer.train(docs, args.vocab)
    tok.save(args.output)
    print(f"vocabulary of {tok.vocab_size} written to {args.output}")
    return 0


def _prepare(args) -> int:
    print(prepare(read_text_docs(args.input, args.separator), Tokenizer.load(args.tokenizer), args.out))
    return 0


def _prepare_chat(args) -> int:
    exclude = evals.chat_set_questions(args.exclude) if args.exclude else frozenset()
    print(prepare_chat(args.input, Tokenizer.load(args.tokenizer), args.out, exclude))
    return 0


def _train(args) -> int:
    model_cfg, train_cfg = load_config(args.config)
    transform = chat_transform(Tokenizer.load(args.tokenizer).special) if args.chat else None
    train(
        model_cfg, train_cfg, args.data, args.run,
        resume=args.resume, init_from=args.init, stop_at=args.stop_at, transform=transform,
    )
    return 0


def _sample(args) -> int:
    model, tok = _load(args)
    new = generate(
        model, tok.encode(args.prompt), args.max_new, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed, stop_id=tok.special["<|eos|>"],
    )
    print(args.prompt + tok.decode(new))
    return 0


def _chat(args) -> int:
    model, tok = _load(args)
    history = []
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if line == "/quit":
            break
        if not line:
            continue
        reply, _ = chat_reply(model, tok, history, line, seed=len(history))
        print(f"model> {reply}")
        history.append((line, reply))
    return 0


def _eval(args) -> int:
    model, tok = _load(args)
    data = Path(args.data)
    val_loss = evals.heldout_loss(model, data / "val.bin", _EVAL_BATCHES, _EVAL_BATCH_SIZE, _EVAL_SEED)
    result = {
        "val_loss": val_loss,
        "perplexity": math.exp(val_loss),
        "unigram_loss": evals.unigram_loss(data / "train.bin", data / "val.bin", tok.vocab_size),
        "samples": evals.run_prompts(model, tok, evals.load_prompts(args.prompts), _EVAL_SEED),
    }
    if args.chat:
        score, details = evals.chat_score(model, tok, evals.load_chat_set(args.chat_set))
        result |= {
            "chat_score": score,
            "chat_stopped": sum(d["stopped"] for d in details) / len(details),
            "chat": details,
        }
    evals.write_result(args.out, result)
    print({k: v for k, v in result.items() if isinstance(v, float)})
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scratchllm")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, handler, summary):
        sub = commands.add_parser(name, help=summary)
        sub.set_defaults(handler=handler)
        return sub

    def model_args(sub):
        sub.add_argument("--ckpt", required=True)
        sub.add_argument("--tokenizer", required=True)

    sub = command("verify", _verify, "hash-check downloaded dataset files")
    sub.add_argument("--registry", default="data/registry.toml")
    sub.add_argument("--raw", default="data/raw")
    sub.add_argument("--id")

    sub = command("licences", _licences, "write the dataset licence table")
    sub.add_argument("--registry", default="data/registry.toml")
    sub.add_argument("--output", default="LICENSES.md")

    sub = command("tokenizer-train", _tokenizer_train, "train the tokenizer on the start of a text file")
    sub.add_argument("--input", required=True)
    sub.add_argument("--separator", required=True)
    sub.add_argument("--vocab", type=int, required=True)
    sub.add_argument("--sample-mb", type=float, default=100)
    sub.add_argument("--output", required=True)

    sub = command("prepare", _prepare, "turn a text file into token files")
    sub.add_argument("--input", required=True)
    sub.add_argument("--separator", required=True)
    sub.add_argument("--tokenizer", required=True)
    sub.add_argument("--out", required=True)

    sub = command("prepare-chat", _prepare_chat, "turn a chat JSON-lines file into token files")
    sub.add_argument("--input", required=True)
    sub.add_argument("--tokenizer", required=True)
    sub.add_argument("--out", required=True)
    sub.add_argument("--exclude", help="chat set whose questions are kept out of training")

    sub = command("train", _train, "train a model")
    sub.add_argument("--config", required=True)
    sub.add_argument("--data", required=True)
    sub.add_argument("--run", required=True)
    sub.add_argument("--resume", action="store_true")
    sub.add_argument("--init", help="checkpoint whose weights start the run")
    sub.add_argument("--chat", action="store_true", help="learn assistant replies only; needs --tokenizer")
    sub.add_argument("--tokenizer")
    sub.add_argument("--stop-at", type=int)

    sub = command("sample", _sample, "continue a prompt")
    model_args(sub)
    sub.add_argument("--prompt", required=True)
    sub.add_argument("--max-new", type=int, default=200)
    sub.add_argument("--temperature", type=float, default=0.8)
    sub.add_argument("--top-p", type=float, default=0.95)
    sub.add_argument("--seed", type=int, default=0)

    sub = command("chat", _chat, "chat in the terminal; /quit to leave")
    model_args(sub)

    sub = command("eval", _eval, "write held-out loss, baseline, samples and (with --chat) the chat score")
    model_args(sub)
    sub.add_argument("--data", required=True)
    sub.add_argument("--out", required=True)
    sub.add_argument("--prompts", default="evals/prompts.txt")
    sub.add_argument("--chat", action="store_true")
    sub.add_argument("--chat-set", default="evals/chat_set.jsonl")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        if args.chat and not args.tokenizer:
            parser.error("--chat needs --tokenizer")
        if args.resume and args.init:
            parser.error("--resume continues a run; --init starts one. Give only one.")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # a character the console cannot show must not stop a run
    try:
        return args.handler(args)
    except (OSError, ValueError, RegistryError, TrainingDiverged) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
