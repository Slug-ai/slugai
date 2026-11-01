# Slug Ai

conftest.py:
Has testing configuration for both controller and host

wire.py:
Has packet networking code shared between both controller and host

/host
Code for each machine to run (profile myself, run a model myself, and pass on the data)

/controller
Code for centeral controller (determine everyones model split based on profiles, send out batches)

# Setup

Setup:
```
uv sync
```

Run all tests:
```
uv run pytest
```

Run a specific thing (ie, text generation of model locally):
Go to folder and read README.md in /host or /controller

# Rules

- NO AI EDITING THE README WITHOUT PERMISSION! NO SLOP READMES!
- Use UV for python