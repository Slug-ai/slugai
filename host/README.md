# Host

Code that runs on hosts, ie, the macbooks, gaming pcs, GPUs, etc.

# Files

profile.py:
Profiles the current host for disk speed, network speed, compute. Useful so we know how much work to shard this node and rebalancing
- Run using ```uv run python -m host.profile mlx-community/Qwen3-8B-4bit --context 256```
- Test using `uv run pytest host/tests/test_profile.py`


model.py:
Runs a range of layers of a model with an option to run using weight streaming
- Run using ```uv run python -m host.model mlx-community/Qwen3-8B-4bit --layers 0:18,18:36 --resident 8```
model is HF repo id
layers is the ranges you want to run
resident is layers in resident
prompt / tokens is the input prompt
- Test using ```uv run pytest host/tests/test_model.py```


host_manager.py:
The host process. Loads a shard when the controller asks, runs each micro-batch sent to it, passes the output to the next host (the last host sends tokens to the controller), and pushes stats every second
- Run using ```uv run python -m host.host_manager --port 47111```
port is where the controller and the previous host connect
- Test using ```uv run pytest host/tests/test_host_manager.py```
Question for later: should we connect from each node to the centeral or to each other directly? should central node be owned by us?