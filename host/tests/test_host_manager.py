import asyncio

import numpy as np

import wire
from host.host_manager import HostManager
from test_model import num_layers

async def start_hosts(n: int) -> list[tuple[HostManager, str]]:
    hosts = []
    for _ in range(n):
        host = HostManager(stats_every=0.05)
        server = await host.serve(0)
        hosts.append((host, f"127.0.0.1:{server.sockets[0].getsockname()[1]}"))
    return hosts


async def recv(reader) -> tuple[dict, np.ndarray | None]:
    """The next message that isn't stats."""
    while True:
        msg, array = await wire.recv(reader)
        if msg["type"] != "stats":
            return msg, array


def test_requests(qwen3):
    async def run():
        ((host, address),) = await start_hosts(1)
        reader, writer = await wire.connect(address)

        async def ask(msg: dict, array: np.ndarray | None = None) -> dict:
            await wire.send(writer, msg, array)
            return (await recv(reader))[0]

        report = await ask({"type": "profile", "model": qwen3, "context": 8, "batch_sizes": [2, 1]})
        assert report["type"] == "profile" and list(report["layer_time"]) == ["1", "2"]
        assert (await ask({"type": "load", "model": qwen3, "start": 0, "end": 2}))["type"] == "error"  # no next host
        assert (await ask({"type": "load", "model": qwen3, "start": 0, "end": num_layers(qwen3)}))["type"] == "loaded"
        error = await ask({"type": "forward", "slot": 5}, np.zeros((1, 1), np.int32))  # never prefilled
        assert error["type"] == "error" and error["slot"] == 5
        stats, _ = await wire.recv(reader)
        assert stats["type"] == "stats" and stats["memory"] > 0
        assert (await ask({"type": "unload"}))["type"] == "unloaded" and host.shard is None
        assert (await ask({"type": "load", "model": qwen3, "start": 0, "end": num_layers(qwen3)}))["type"] == "loaded"
        writer.close()  # the controller leaving unloads the shard
        while host.shard is not None:
            await asyncio.sleep(0.01)

    asyncio.run(asyncio.wait_for(run(), 30))


def test_frames_from_an_old_plan_are_dropped(qwen3):
    async def run():
        ((_, address),) = await start_hosts(1)
        reader, writer = await wire.connect(address)
        await wire.send(writer, {"type": "load", "model": qwen3, "start": 0, "end": num_layers(qwen3), "epoch": 2})
        assert (await recv(reader))[0]["type"] == "loaded"
        x = np.array([[1, 2, 3]], np.int32)
        await wire.send(writer, {"type": "forward", "slot": 0, "lengths": [3], "epoch": 1}, x)
        await wire.send(writer, {"type": "forward", "slot": 0, "lengths": [3], "epoch": 2}, x)
        msg, _ = await recv(reader)
        assert msg["type"] == "tokens" and msg["epoch"] == 2

    asyncio.run(asyncio.wait_for(run(), 30))
