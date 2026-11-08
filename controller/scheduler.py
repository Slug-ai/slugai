"""Hands prompts out to slots (micro-batches in flight) and keeps every token generated so far.

Pure: it returns the frames to send to the first host, and the controller does
the sending. A micro-batch starts with two frames: a prefill of all but each
sequence's last token, right-padded, then a decode step with the last tokens,
so the rows line up. Each decode step's tokens give the next step's input
until every sequence in the slot has ended (at an end-of-sequence token or
``max_tokens``); then the slot starts the next queued sequences.

Because the scheduler holds every token, a slot can be started again on a new
plan: a sequence's context is its prompt plus what it has generated so far.
"""

from __future__ import annotations

import numpy as np
from slug_api import Queue, Sequence

Frame = tuple[dict, np.ndarray]  # a forward message and its token ids


class Scheduler(Queue):
    """A ``Queue`` (requests in line, prompts given here first) that also hands them out to slots."""

    def __init__(self, prompts: list[list[int]], max_tokens: int, eos: set[int]):
        super().__init__(max_tokens, eos)
        for prompt in prompts:
            self.add(prompt)
        self.slots: dict[int, list[Sequence]] = {}
        self.widths: dict[int, int] = {}  # slot -> sequences it takes

    @property
    def finished(self) -> bool:
        return not self.waiting and not self.slots

    def start(self, slot: int, batch: int) -> list[Frame]:
        """Put up to ``batch`` queued sequences in ``slot``: the frames that start it (none if the queue is empty)."""
        self.widths[slot] = batch
        seqs = [self.waiting.popleft() for _ in range(min(batch, len(self.waiting)))]
        if not seqs:
            return []
        self.slots[slot] = seqs
        contexts = [s.prompt + s.tokens for s in seqs]
        lengths = [len(c) - 1 for c in contexts]
        width = max(1, *lengths)
        prefill = np.array([c[:-1] + [0] * (width - n) for c, n in zip(contexts, lengths)], np.int32)
        last = np.array([[c[-1]] for c in contexts], np.int32)
        return [({"type": "forward", "slot": slot, "lengths": lengths}, prefill), ({"type": "forward", "slot": slot}, last)]

    def advance(self, slot: int, tokens: list[int]) -> list[Frame]:
        """Record a decode step's ``tokens`` (one per row of ``slot``): the frames to send next."""
        seqs = self.slots[slot]
        for s, t in zip(seqs, tokens):
            if not s.done:
                s.tokens.append(t)
                s.done = t in self.eos or len(s.tokens) >= s.max_tokens
        if not all(s.done for s in seqs):
            return [({"type": "forward", "slot": slot}, np.array(tokens, np.int32)[:, None])]
        del self.slots[slot]
        return self.start(slot, self.widths[slot])

    def requeue(self) -> None:
        """Put every unfinished sequence in a slot back at the front of the queue, keeping its tokens (for a new plan)."""
        running = [s for seqs in self.slots.values() for s in seqs if not s.done]
        self.waiting.extendleft(reversed(sorted(running, key=lambda s: s.id)))
        self.slots.clear()
