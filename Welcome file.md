## DRL implementation
An cleanRL style implementation of gail and dagger


## Clone Repository 
First, clone the repository to your local machine: 
```bash 
git clone https://github.com/abstcol/imitation_learning.git
cd imitation_learing
```


## Installation
Set up the project environment with:

```bash
conda create -n imitationl python=3.10  
conda activate imitationl 
pip install -r requirements.txt
```
If you encounter package conflicts, try installing dependencies manually.


## Training
You can train the agent with 
```bash
python gail.py
```
(embarrasingly, I can't find a suitable hyperparameters for gail, so it works indeed badly. However, the gail_dagger works very well! So you can start with it.)
It's noteworthing that all the code needed to train the agent(include the expert experience produce) is in one script. So it's super easy to understand for fresh learner.

## Logs&Checkpoints

Training logs and checkpoints will be saved in wandb after experiement.
So please make sure that you log in your wandb account before experiement.

##  Produce video
You can create a video of the agent's performance. Simply set the following config:
```python
capture_video:bool=True
    """whether to capture videos of the agent performances (check out 'videos' folder)"""
```
This will generate a video showcasing the agent's behavior during testing. 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNzgwNTQ0NzldfQ==
-->