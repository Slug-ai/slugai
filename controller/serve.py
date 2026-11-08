"""Serve a model across hosts with slug-api's OpenAI-compatible API: the controller in serving mode is the backend.

    uv run python -m controller.serve --model mlx-community/Qwen3-8B-4bit --hosts mac1:47111,mac2:47111 --key <key>
"""

from __future__ import annotations

import argparse

from slug_api import serve

from .controller import Controller, encoder, load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Serve a model across hosts with an OpenAI-compatible API.")
    p.add_argument("--model", required=True, help="Hugging Face repo id or local model directory (the same on every host)")
    p.add_argument("--hosts", required=True, help="host:port of each host, in pipeline order, e.g. mac1:47111,mac2:47111")
    p.add_argument("--max-tokens", type=int, default=512, help="tokens to generate for a request that doesn't say")
    p.add_argument("--context", type=int, default=4096, help="most tokens a request can have, prompt plus completion")
    p.add_argument("--slots", type=int, help="micro-batches in flight (default: one per host, plus one)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--bind", default="127.0.0.1", help="address to listen on (0.0.0.0 for every interface)")
    p.add_argument("--key", help="require Authorization: Bearer <key> on every request")
    args = p.parse_args()

    tok, eos = load_tokenizer(args.model)
    controller = Controller(args.model, args.hosts.split(","), [], args.max_tokens, eos, args.slots, context=args.context)
    serve(controller, encoder(tok), lambda t: tok.decode(t, skip_special_tokens=True), args.model, args.key, args.bind, args.port)


if __name__ == "__main__":
    main()
