# Controller

Code that runs the job on a centeral control plane.
It will talks to every host, plans the split of layers, feeds prompts through and collects completions.

# Files

controller.py:
Runs the entire job e2e (profile + plan + load plan to host + stream prompts + save outputs) 
- Run using ```uv run python -m controller.controller --model mlx-community/Qwen3-8B-4bit --hosts mac1:47111,mac2:47111 --prompts prompts.txt --max-tokens 128```
model is HF repo id (the same on every host)
hosts is host:port of each host, in pipeline order
prompts is a .txt with one prompt per line, or .jsonl with {"prompt": ...} per line
max-tokens is the most tokens to generate per prompt
slots is micro-batches in flight (default: one per host, plus one)
job names the output files
- Test using ```uv run pytest controller/tests/test_controller.py```


plan.py:
Takes the model size and each host's profile (from the profiling) and decides the split: which layers go on which host, how many stay in memory (the rest stream from disk), the micro-batch size and how many slots. Picks the split where the slowest host is fastest. Doesn't account for network speed yet
- Test using ```uv run pytest controller/tests/test_plan.py```


scheduler.py:
Prompt queue and slots. Owns every token generated so far, so sequences pick up where they were after a replan
- Test using ```uv run pytest controller/tests/test_scheduler.py```
