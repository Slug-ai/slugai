"""The controller as slug-api's backend, through the openai package: real hosts, mlx-lm's tokens."""

import asyncio
import json

import httpx
import openai
from slug_api import Server

from controller.controller import Controller
from controller.tests.test_controller import EOS, PROMPTS, reference, start_hosts

MODEL = "slug/tiny"


def encode(messages: list[dict]) -> list[int]:
    return [int(t) for t in messages[-1]["content"].split()]


def decode(tokens: list[int]) -> str:
    return " ".join(map(str, tokens))


def ask(prompt: list[int]) -> list[dict]:
    return [{"role": "user", "content": decode(prompt)}]


async def serve(controller: Controller, test):
    """Run ``test(client)`` with an ``openai.AsyncOpenAI`` pointed at the API, the controller running."""
    server = Server(controller, encode, decode, MODEL)
    async with server.app.router.lifespan_context(server.app):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(server.app))
        client = openai.AsyncOpenAI(base_url="http://api/v1", api_key="none", http_client=http, max_retries=0)
        return await test(client)


def test_chat_completions_streamed_or_not_and_batches_give_mlx_lms_tokens(qwen3):
    async def run():
        _, addresses = await start_hosts(2)
        controller = Controller(qwen3, addresses, [], max_tokens=6, eos=EOS, batch_sizes=[1, 2, 4], context=64)

        async def test(client: openai.AsyncOpenAI):
            lines = [{"custom_id": f"r{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MODEL, "messages": ask(p)}} for i, p in enumerate(PROMPTS)]
            file = await client.files.create(file=("in.jsonl", "".join(json.dumps(l) + "\n" for l in lines).encode()), purpose="batch")
            batch = await client.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
            single = await client.chat.completions.create(model=MODEL, messages=ask(PROMPTS[0]))
            stream = await client.chat.completions.create(model=MODEL, messages=ask(PROMPTS[1]), stream=True, stream_options={"include_usage": True})
            chunks = [c async for c in stream]
            batch = await client.batches.retrieve(batch.id, extra_query={"wait": 30})
            output = (await client.files.content(batch.output_file_id)).text
            return single, chunks, batch, output

        return await serve(controller, test)

    single, chunks, batch, output = asyncio.run(asyncio.wait_for(run(), 60))
    expected = [decode(t) for t in reference(qwen3, 6)]
    reasons = ["stop" if t[-1] in EOS else "length" for t in reference(qwen3, 6)]

    assert single.choices[0].message.content == expected[0] and single.choices[0].finish_reason == reasons[0]
    assert single.usage.prompt_tokens == len(PROMPTS[0]) and single.model == MODEL

    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == expected[1]
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == reasons[1]
    assert chunks[-1].usage.completion_tokens == len(reference(qwen3, 6)[1])

    assert batch.status == "completed" and batch.request_counts.completed == len(PROMPTS)
    results = {r["custom_id"]: r["response"]["body"]["choices"][0]["message"]["content"] for r in map(json.loads, output.splitlines())}
    assert results == {f"r{i}": e for i, e in enumerate(expected)}
