# gym-numgrid

The NumGrid environment consists of a grid of hand-written digits images loaded from a MNIST database file in IDX gzipped format, as can be found on [LeCun's website](http://yann.lecun.com/exdb/mnist/).

The environment holds a cursor representing the agent's local view on the world (aka the grid); the cursor can either be moved on a small distance in one of the 4 orthogonal directions, or be directly teleported at a given position (the latter constituting a substantially larger action space).

The agent's goal is to reach the highest possible speed in accurately guessing the digit it is currently viewing. Right labelling leads to a positive reward, and wrong labelling to a negative one. The agent can of course take some steps to prepare its answer by exploring the image, in which case it can label with a 10 to tell the environment to ignore the answer.

A part of the task's difficulty resides in the agent's ability to keep the cursor focused on the same digit to avoid mixing information from different images, which would be confusing.

# Installation

```bash
cd gym-numgrid
pip install -e .
```

To get started with the environment, you can run it with one of the agents in `examples/agents`:
```bash
python examples/agents/random_agent.py
```
