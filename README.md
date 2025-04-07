
## DRL Implementation

A CleanRL-style implementation of GAIL and DAgger.

## Clone Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/abstcol/imitation_learning.git
cd imitation_learning

```

## Installation

Set up the project environment with:

```bash
conda create -n imitationl python=3.10
conda activate imitationl
pip install -r requirements.txt

```

If you encounter package conflicts, try installing the dependencies manually.

## Training

You can train the agent with the following command:

```bash
python gail.py

```

Note: Finding suitable hyperparameters for GAIL has proven challenging, and its performance is currently suboptimal. However, the GAIL+DAgger implementation performs significantly better and is recommended as a starting point.

All the code necessary for training the agent (including generating expert demonstrations) is contained in a single script. This design makes it particularly easy for beginners to understand and follow.

## Logs & Checkpoints

Training logs and checkpoints will be saved to Weights & Biases (wandb) after each experiment. Please ensure that you are logged into your wandb account before starting an experiment.

## Produce Video

To generate a video of the agent's performance, set the following configuration:

```python
capture_video: bool = True
"""Whether to capture videos of the agent's performance (check the 'videos' folder)."""

```

This will produce a video that showcases the agent's behavior during testing.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTkxMzI0MTMzM119
-->