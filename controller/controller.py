"""Run a job across hosts: profile them, plan the split, load it, stream the prompts through and collect completions.

The controller holds a connection to every host (host/host_manager.py lists
the messages). The scheduler feeds micro-batches to the first host; the last
host sends back each step's tokens, which the scheduler turns into the next
step's input. When a host drops (its connection closes or its stats stop),
the controller plans again on the hosts left, loads that plan under a new
epoch (hosts drop frames from the old one) and restarts the slots from the
tokens the scheduler holds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import wire

from .plan import HostProfile, ModelSize, Plan, plan
from .scheduler import Frame, Scheduler, Sequence


class Link:
    """The connection to one host: requests wait for their replies; tokens, stats and errors go to ``events``."""

    def __init__(self, host: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, events: asyncio.Queue):
        self.host, self.writer = host, writer
        self.replies: asyncio.Queue = asyncio.Queue()
        self.last_seen = time.monotonic()  # when the host last sent anything (its stats come every second)
        self.reading = asyncio.create_task(self.read(reader, events))

    async def read(self, reader: asyncio.StreamReader, events: asyncio.Queue) -> None:
        try:
            while True:
                msg, array = await wire.recv(reader)
                self.last_seen = time.monotonic()
                if msg["type"] in ("tokens", "stats") or "slot" in msg:  # a micro-batch's error has its slot
                    events.put_nowait((self.host, msg, array))
                else:
                    self.replies.put_nowait(msg)
        except (asyncio.IncompleteReadError, ConnectionError):
            events.put_nowait((self.host, {"type": "closed"}, None))
            self.replies.put_nowait({"type": "error", "message": "connection closed"})

    async def ask(self, msg: dict) -> dict:
        await wire.send(self.writer, msg)
        reply = await self.replies.get()
        if reply.pop("type") == "error":
            raise RuntimeError(f"{self.host}: {reply['message']}")
        return reply

    def close(self) -> None:
        self.reading.cancel()
        self.writer.close()


class Controller:
    def __init__(
        self,
        model: str,
        hosts: list[str],
        prompts: list[list[int]],
        max_tokens: int,
        eos: set[int],
        slots: int | None = None,
        batch_sizes: list[int] | None = None,
        timeout: float = 10.0,
    ):
        """``hosts``: ``host:port`` each, in pipeline order. ``slots``: micro-batches in flight (default: one per host, plus one).
        ``batch_sizes``: what hosts profile (default: theirs). ``timeout``: seconds without a word before a host has dropped."""
        self.model, self.hosts, self.max_tokens, self.slots, self.batch_sizes, self.timeout = model, hosts, max_tokens, slots, batch_sizes, timeout
        self.scheduler = Scheduler(prompts, max_tokens, eos)
        self.events: asyncio.Queue = asyncio.Queue()  # (host, message, array) from every link
        self.links: dict[str, Link] = {}
        self.stats: dict[str, dict] = {}  # host -> its latest stats
        self.epoch = 0
        self.plan: Plan | None = None
        self.started = time.monotonic()

    async def run(self, status_every: float = 5.0) -> list[Sequence]:
        """Run the job to the end, printing its status as it goes: every sequence and its tokens, in prompt order."""
        for host in self.hosts:
            self.links[host] = Link(host, *await wire.connect(host), self.events)
        await self.measure()
        await self.apply()
        self.started = shown = time.monotonic()  # time the job from here, after profiling and loading
        while not self.scheduler.finished:
            try:
                await self.handle(*await asyncio.wait_for(self.events.get(), self.timeout / 4))
            except TimeoutError:
                pass
            while silent := [h for h, link in self.links.items() if time.monotonic() - link.last_seen > self.timeout]:
                await self.drop(silent[0])  # one at a time: planning again hears from the others
            if time.monotonic() - shown > status_every:
                print(self.status(), flush=True)
                shown = time.monotonic()
        for link in self.links.values():
            link.close()  # hosts unload when the controller leaves
        return self.scheduler.sequences

    async def measure(self) -> None:
        """The model's sizes (from the first host) and every host's profile, at the job's typical context."""
        first = self.links[self.hosts[0]]
        self.size = ModelSize(**await first.ask({"type": "size", "model": self.model}))
        lengths = [len(s.prompt) for s in self.scheduler.sequences]
        ask = {"type": "profile", "model": self.model, "context": int(statistics.mean(lengths)) + self.max_tokens // 2, "batch_sizes": self.batch_sizes}
        self.profiles = {}
        for host, link in self.links.items():  # one at a time: hosts sharing a Mac would skew each other's
            p = await link.ask(ask)
            self.profiles[host] = HostProfile({int(b): t for b, t in p["layer_time"].items()}, p["free_memory"], p["disk_speed"])
        self.context = max(lengths) + self.max_tokens  # the longest any cache gets

    async def apply(self) -> None:
        """Plan on the hosts left, load the plan under a new epoch and start every slot."""
        if self.scheduler.finished:
            return
        self.epoch += 1
        slots = min(self.slots or len(self.links) + 1, len(self.scheduler.queue))
        self.plan = plan(self.size, {h: self.profiles[h] for h in self.links}, self.context, slots, math.ceil(len(self.scheduler.queue) / slots))
        stages = self.plan.stages
        hosts = ", ".join(f"{s.host} layers {s.start}-{s.end - 1} ({s.resident} in memory)" for s in stages)
        print(f"plan {self.epoch}: {self.plan.slots} x {self.plan.batch} sequences, ~{self.plan.tokens_per_second:.1f} tok/s; {hosts}", flush=True)
        await asyncio.gather(*(
            self.links[s.host].ask({"type": "load", "model": self.model, "start": s.start, "end": s.end, "resident": s.resident, "next": nxt.host if nxt else None, "epoch": self.epoch})
            for s, nxt in zip(stages, stages[1:] + [None])
        ))
        self.first = self.links[stages[0].host]
        for slot in range(self.plan.slots):
            await self.feed(self.scheduler.start(slot, self.plan.batch))

    async def handle(self, host: str, msg: dict, array: np.ndarray | None) -> None:
        if msg["type"] == "stats":
            self.stats[host] = msg
        elif msg["type"] == "closed":
            await self.drop(host)
        elif msg["type"] == "error":
            raise RuntimeError(f"{host}: {msg['message']}")
        elif msg.get("epoch") == self.epoch and "lengths" not in msg:  # a prefill's tokens follow cut-short prompts
            await self.feed(self.scheduler.advance(msg["slot"], array.tolist()))

    async def feed(self, frames: list[Frame]) -> None:
        try:
            for msg, x in frames:
                await wire.send(self.first.writer, {**msg, "epoch": self.epoch}, x)
        except ConnectionError:
            pass  # the first host is gone: it's about to drop

    async def drop(self, host: str) -> None:
        if host not in self.links:
            return
        print(f"{host} dropped; planning again without it", flush=True)
        self.links.pop(host).close()
        self.stats.pop(host, None)
        self.scheduler.requeue()
        await self.apply()

    def status(self) -> str:
        seqs = self.scheduler.sequences
        done, tokens = sum(s.done for s in seqs), sum(len(s.tokens) for s in seqs)
        hosts = "  ".join(f"{h} {s['busy']:.0%} busy {s['memory'] / 1e9:.1f} GB" for h, s in self.stats.items())
        return f"{done}/{len(seqs)} done  {tokens / (time.monotonic() - self.started):.1f} tok/s  {hosts}"


def save(job: str, controller: Controller, decode: Callable[[list[int]], str]) -> list[Path]:
    """Write ``outputs/<job>-completions.jsonl`` (a line per prompt) and ``outputs/<job>-report.json``."""
    out, seqs = Path("outputs"), controller.scheduler.sequences
    out.mkdir(exist_ok=True)
    completions, report = out / f"{job}-completions.jsonl", out / f"{job}-report.json"
    completions.write_text("".join(json.dumps({"id": s.id, "completion": decode(s.tokens), "tokens": s.tokens}) + "\n" for s in seqs))
    seconds, tokens = time.monotonic() - controller.started, sum(len(s.tokens) for s in seqs)
    summary = {"plan": asdict(controller.plan), "seconds": seconds, "tokens": tokens, "tokens_per_second": tokens / seconds, "hosts": controller.stats}
    report.write_text(json.dumps(summary, indent=1))
    return [completions, report]


def read_prompts(path: str) -> list[str]:
    """One prompt per line of a .txt, or a ``{"prompt": ...}`` per line of a .jsonl."""
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    return [json.loads(line)["prompt"] for line in lines] if path.endswith(".jsonl") else lines


def load_tokenizer(model: str):
    """The model's tokenizer and end-of-sequence ids, fetching only its small files."""
    path = Path(model) if Path(model).exists() else Path(snapshot_download(model, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"]))
    tok = AutoTokenizer.from_pretrained(path)
    eos = json.loads((path / "config.json").read_text()).get("eos_token_id")
    return tok, {tok.eos_token_id, *(eos if isinstance(eos, list) else [eos])} - {None}


def main() -> None:
    p = argparse.ArgumentParser(description="Run a job across hosts and write its completions and report to outputs/.")
    p.add_argument("--model", required=True, help="Hugging Face repo id or local model directory (the same on every host)")
    p.add_argument("--hosts", required=True, help="host:port of each host, in pipeline order, e.g. mac1:47111,mac2:47111")
    p.add_argument("--prompts", required=True, help='.txt with one prompt per line, or .jsonl with {"prompt": ...} per line')
    p.add_argument("--max-tokens", type=int, default=128, help="tokens to generate per prompt, at most")
    p.add_argument("--slots", type=int, help="micro-batches in flight (default: one per host, plus one)")
    p.add_argument("--job", default=time.strftime("job-%Y%m%d-%H%M%S"), help="names the output files")
    args = p.parse_args()

    tok, eos = load_tokenizer(args.model)

    def encode(text: str) -> list[int]:  # instruct models get their chat template
        if tok.chat_template:
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            return tok.encode(text, add_special_tokens=False)
        return tok.encode(text)

    prompts = [encode(t) for t in read_prompts(args.prompts)]
    controller = Controller(args.model, args.hosts.split(","), prompts, args.max_tokens, eos, args.slots)
    asyncio.run(controller.run())
    print(controller.status())
    for path in save(args.job, controller, lambda t: tok.decode(t, skip_special_tokens=True)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
