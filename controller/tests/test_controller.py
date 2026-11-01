import asyncio
import json

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from controller.controller import Controller, save
from controller.plan import BLOCK
from host.host_manager import HostManager
from host.model import model_path, size

PROMPTS = [[5, 9, 3, 7, 1, 12], [4, 2], [8, 8, 30, 2, 2, 11], [6], [13, 14, 15], [40, 41, 42, 43, 44, 45, 46], [9, 9]]
EOS = {2}


async def start_hosts(n: int) -> tuple[list[HostManager], list[str]]:
    hosts = [HostManager(stats_every=0.05) for _ in range(n)]
    servers = [await h.serve(0) for h in hosts]
    return hosts, [f"127.0.0.1:{s.sockets[0].getsockname()[1]}" for s in servers]


def reference(model: str, max_tokens: int) -> list[list[int]]:
    """mlx-lm's greedy tokens for each prompt alone, up to an end-of-sequence token or ``max_tokens``."""
    m, _ = load_model(model_path(model))
    out = []
    for prompt in PROMPTS:
        cache, x, tokens = make_prompt_cache(m), mx.array([prompt]), []
        while len(tokens) < max_tokens and (not tokens or tokens[-1] not in EOS):
            x = mx.argmax(m(x, cache=cache)[:, -1], -1)[:, None]
            tokens.append(x.item())
        out.append(tokens)
    return out


def test_job_matches_mlx_lm(tiny_model, tmp_path, monkeypatch):
    async def run():
        _, addresses = await start_hosts(3)
        controller = Controller(tiny_model, addresses, PROMPTS, max_tokens=6, eos=EOS, batch_sizes=[1, 2, 4])
        return controller, await controller.run()

    controller, sequences = asyncio.run(asyncio.wait_for(run(), 60))
    assert [s.tokens for s in sequences] == reference(tiny_model, 6)
    assert len(controller.plan.stages) == 3 and controller.status().startswith(f"{len(PROMPTS)}/{len(PROMPTS)} done")
    monkeypatch.chdir(tmp_path)
    completions, report = save("job", controller, decode=lambda tokens: " ".join(map(str, tokens)))
    assert [json.loads(line)["tokens"] for line in completions.read_text().splitlines()] == [s.tokens for s in sequences]
    assert json.loads(report.read_text())["tokens"] == sum(len(s.tokens) for s in sequences)


def test_a_model_too_big_for_its_host_streams(tiny_model, monkeypatch):
    """The host says it has room for the caches and three layers: one to keep, two to stream the rest through."""
    s = size(tiny_model)
    caches = s["layers"] * (s["kv_bytes"] * BLOCK + s["state_bytes"])  # one sequence at a time; these jobs fit one block
    free = s["embed_bytes"] + s["head_bytes"] + caches + 3 * s["layer_bytes"]
    monkeypatch.setattr("host.profile.free_memory", lambda: free)

    async def run():
        _, addresses = await start_hosts(1)
        controller = Controller(tiny_model, addresses, PROMPTS, max_tokens=6, eos=EOS, slots=1, batch_sizes=[1, 2, 4])
        return controller, await controller.run()

    controller, sequences = asyncio.run(asyncio.wait_for(run(), 60))
    (stage,) = controller.plan.stages
    assert stage.resident < s["layers"]  # the plan streams the layers that don't fit
    assert [q.tokens for q in sequences] == reference(tiny_model, 6)


def test_a_host_that_stops_is_planned_around(tiny_model):
    async def run():
        hosts, addresses = await start_hosts(3)
        controller = Controller(tiny_model, addresses, PROMPTS, max_tokens=12, eos=EOS, batch_sizes=[1, 2, 4], timeout=0.5)

        async def hang_the_middle_host():
            while sum(len(s.tokens) for s in controller.scheduler.sequences) < 5:
                await asyncio.sleep(0.001)
            for task in hosts[1].tasks:  # it stops computing, forwarding and pushing stats
                task.cancel()

        hang = asyncio.create_task(hang_the_middle_host())
        sequences = await controller.run()
        hang.cancel()
        return controller, addresses, sequences

    controller, addresses, sequences = asyncio.run(asyncio.wait_for(run(), 60))
    assert controller.epoch == 2 and [s.host for s in controller.plan.stages] == [addresses[0], addresses[2]]
    assert [s.tokens for s in sequences] == reference(tiny_model, 12)


def test_tokens_from_an_old_plan_are_ignored():
    async def run():
        controller = Controller("model", ["a:1"], [[1, 2]], max_tokens=4, eos=EOS)
        controller.epoch = 2
        controller.scheduler.start(0, 1)
        await controller.handle("a:1", {"type": "tokens", "slot": 0, "epoch": 1}, np.array([7]))
        return controller.scheduler.sequences[0].tokens

    assert asyncio.run(run()) == []
