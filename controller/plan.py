"""Split a model's layers across hosts from their profiles: pure arithmetic, so it's fast and testable.

Hosts run in the order given, each holding a contiguous block of at least one
layer; the first also holds the embedding and the last the head. With ``k``
layers, a host spends ``k * t(b)`` seconds on a micro-batch of ``b`` sequences
(``t`` is its profiled time per layer) and needs room for those layers'
weights and for the caches of every sequence in flight: ``slots * b`` of them,
``context`` tokens each. Layers that don't fit stream from disk through two
layers' worth of room, each adding the part of its read that compute doesn't
hide: ``max(0, layer_bytes / disk_speed - t(b))``.

The pipeline moves at its slowest host, so for each micro-batch size the plan
finds the split that makes that host fastest, then keeps the size that gives
the most tokens per second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BLOCK = 256  # caches grow in blocks of this many tokens


@dataclass
class ModelSize:  # bytes, as a host's ``size`` reply gives them
    layers: int
    layer_bytes: int  # weights per layer
    embed_bytes: int  # held by the first host
    head_bytes: int  # final norm and lm_head, held by the last host
    kv_bytes: int  # cache per layer per token
    state_bytes: int  # cache per layer per sequence (linear attention's state)


@dataclass
class HostProfile:  # as a host's ``profile`` reply gives it
    layer_time: dict[int, float]  # micro-batch size -> seconds per layer per decode step
    free_memory: int  # bytes
    disk_speed: float  # bytes/s


@dataclass
class Stage:
    host: str
    start: int
    end: int
    resident: int  # layers kept in memory; the rest stream from disk


@dataclass
class Plan:
    stages: list[Stage]
    batch: int  # sequences per micro-batch
    slots: int  # micro-batches in flight
    step_time: float  # seconds per micro-batch on the slowest host

    @property
    def tokens_per_second(self) -> float:
        return self.batch / self.step_time


class PlanError(ValueError):
    pass


def plan(size: ModelSize, hosts: dict[str, HostProfile], context: int, slots: int, max_batch: int) -> Plan:
    """The fastest plan for ``slots`` micro-batches of at most ``max_batch`` sequences, ``context`` tokens each."""
    if not 0 < len(hosts) <= size.layers:
        raise PlanError(f"need 1 to {size.layers} hosts, got {len(hosts)}")
    profiled = sorted(set.intersection(*(set(h.layer_time) for h in hosts.values())))
    batches = {b for b in profiled if b < max_batch} | {min(max_batch, profiled[-1])}
    context = math.ceil(context / BLOCK) * BLOCK
    plans = [p for b in sorted(batches) if (p := _split(size, hosts, context, slots, b))]
    if not plans:
        raise PlanError("the caches don't fit, even streaming every layer from disk; use fewer slots or tokens")
    return max(plans, key=lambda p: p.tokens_per_second)


def _split(size: ModelSize, hosts: dict[str, HostProfile], context: int, slots: int, b: int) -> Plan | None:
    """The split that makes the slowest host fastest for micro-batches of ``b`` (None: nothing fits)."""
    names, L = list(hosts), size.layers
    extra = lambda i: (size.embed_bytes if i == 0 else 0) + (size.head_bytes if i == len(names) - 1 else 0)
    # costs[i][k - 1]: (seconds, resident layers) for k layers on host i; seconds grow with k
    costs = [[_stage(size, hosts[h], k, b, slots, context, extra(i)) for k in range(1, L + 1)] for i, h in enumerate(names)]
    # the slowest host's time is one of these: take the smallest that leaves room for every layer
    for limit in sorted({t for row in costs for t, _ in row if t < math.inf}):
        most = [sum(t <= limit for t, _ in row) for row in costs]  # layers each host can take within it
        if min(most) >= 1 and sum(most) >= L:
            break
    else:
        return None
    stages, start = [], 0
    for i, (name, row) in enumerate(zip(names, costs)):
        k = min(most[i], L - start - (len(names) - 1 - i))  # leave at least one layer for each host after
        stages.append(Stage(name, start, start + k, row[k - 1][1]))
        start += k
    step_time = max(row[s.end - s.start - 1][0] for row, s in zip(costs, stages))
    return Plan(stages, b, slots, step_time)


def _stage(size: ModelSize, host: HostProfile, layers: int, b: int, slots: int, context: int, extra: int) -> tuple[float, int]:
    """Seconds per micro-batch and layers kept in memory for ``layers`` layers on ``host`` (inf: they don't fit)."""
    sizes = sorted(host.layer_time)
    t = float(np.interp(b, sizes, [host.layer_time[s] for s in sizes]))
    w = size.layer_bytes
    room = host.free_memory - extra - layers * slots * b * (size.kv_bytes * context + size.state_bytes)
    if room >= layers * w:
        return layers * t, layers
    resident = (room - 2 * w) // w  # streamed layers pass through room for two: the one in use and the next
    if resident < 0 or host.disk_speed <= 0:
        return math.inf, 0
    return layers * t + (layers - resident) * max(0.0, w / host.disk_speed - t), resident
