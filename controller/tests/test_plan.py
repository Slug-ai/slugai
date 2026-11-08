import pytest

from controller.plan import BLOCK, HostProfile, ModelSize, Plan, PlanError, plan

GB, MB = 1 << 30, 1 << 20
SIZE = ModelSize(layers=36, layer_bytes=100 * MB, embed_bytes=300 * MB, head_bytes=300 * MB, kv_bytes=4096, state_bytes=0)


def host(ms: float = 1.0, free: float = 16, disk: float = 5) -> HostProfile:
    """``ms`` per layer for one sequence, a little more for bigger batches; ``free`` GB; ``disk`` GB/s."""
    return HostProfile({b: ms * 1e-3 * (1 + b / 64) for b in (1, 8, 32)}, int(free * GB), disk * GB)


def layers(p: Plan) -> list[int]:
    return [s.end - s.start for s in p.stages]


def check(p: Plan, hosts: dict[str, HostProfile], context: int) -> None:
    """Every layer once, in order, and every host within its memory."""
    assert [s.host for s in p.stages] == list(hosts)
    assert [s.start for s in p.stages] == [0] + [s.end for s in p.stages[:-1]] and p.stages[-1].end == SIZE.layers
    context = -(-context // BLOCK) * BLOCK
    for i, s in enumerate(p.stages):
        k, w = s.end - s.start, SIZE.layer_bytes
        held = s.resident * w + (2 * w if s.resident < k else 0) + k * p.slots * p.batch * SIZE.kv_bytes * context
        held += (SIZE.embed_bytes if i == 0 else 0) + (SIZE.head_bytes if i == len(p.stages) - 1 else 0)
        assert held <= hosts[s.host].free_memory


def test_equal_hosts_split_evenly():
    hosts = {"a": host(), "b": host()}
    p = plan(SIZE, hosts, context=300, slots=3, max_batch=32)
    check(p, hosts, 300)
    assert layers(p) == [18, 18] and p.stages[0].resident == 18


def test_a_faster_host_gets_more_layers():
    hosts = {"fast": host(ms=1), "slow": host(ms=2)}
    p = plan(SIZE, hosts, context=300, slots=3, max_batch=32)
    check(p, hosts, 300)
    assert layers(p) == [24, 12]


def test_memory_caps_a_host():
    hosts = {"a": host(), "small": host(free=1.3)}
    p = plan(SIZE, hosts, context=300, slots=3, max_batch=8)
    check(p, hosts, 300)
    assert layers(p)[1] < 18 and p.stages[1].resident == layers(p)[1]


def test_streams_the_layers_that_dont_fit():
    hosts = {"a": host(free=2.5)}  # the model takes 4.1 GB
    p = plan(SIZE, hosts, context=300, slots=1, max_batch=1)
    check(p, hosts, 300)
    streamed = 36 - p.stages[0].resident
    assert streamed > 0
    assert p.step_time == pytest.approx(36 * 1e-3 * (1 + 1 / 64) + streamed * (100 * MB / (5 * GB) - 1e-3 * (1 + 1 / 64)))


def test_picks_the_batch_with_the_most_tokens_per_second():
    hosts = {"a": host(), "b": host()}
    assert plan(SIZE, hosts, context=300, slots=3, max_batch=32).batch == 32
    assert plan(SIZE, hosts, context=300, slots=3, max_batch=5).batch == 5  # time per layer interpolated


def test_caches_that_cant_fit():
    with pytest.raises(PlanError):
        plan(SIZE, {"a": host(free=4)}, context=100_000, slots=3, max_batch=32)
