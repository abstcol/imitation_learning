from dataclasses import dataclass
import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset,DataLoader
from torch import optim
from torch.distributions.categorical import Categorical
import tyro
import time
from torch.utils.tensorboard import SummaryWriter
import random
from typing import Callable
import torch.nn.functional as F
import datetime



torch.autograd.set_detect_anomaly(True)
@dataclass
class Args:
    exp_name:str=os.path.basename(__file__)[:-len(".py")]
    """the name of this experiment"""
    seed:int =1
    """seed of the experiment"""
    torch_deterministic:bool=True
    """if toggled, 'torch.backends.cudnn.deterministic=False'"""
    cuda:bool=True
    """if toggled, cuda will be enabled by default"""
    track:bool=True
    """if toggled, this experiment will be tracked with w&b"""
    wandb_project_name:str="imitation_learning"
    """the wandb's project name"""
    wandb_entity:str=None
    """the entity(team) of wandb's project"""
    capture_video:bool=True
    """whether to capture videos of the agent performances (check out 'videos' folder)"""

    env_id:str="CartPole-v1"
    """the if of the environment"""
    total_timesteps:int =500000
    """total timesteps of the experiments"""
    learning_rate:float=5e-4
    """the learning rate of the optimizer"""
    num_envs:int=4
    """the number of parallel game environments"""
    num_steps:int=128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr:bool=True
    """Toggle learning rate annealing for policy and value networks"""
    gamma:float=0.99
    """the discounter factor gamma"""
    gae_lambda:float=0.95
    """the lambda for the general advantage estimate"""
    num_minibatches:int =4
    """the number of mini-batches"""
    update_epochs:int =1
    """the K epochs to update the policy"""
    ent_coef:float=0.01
    """coefficient of the entropy"""
    max_grad_norm:float=1
    """"""
    clip_coef:float=0.2
    """the surrogate clipping coefficient"""
    clip_vloss:bool=True
    """toggles whether or not to use a clipped loss for the value function"""
    vf_coef:float=0.1
    """coefficient of the value functoin"""
    norm_adv:bool=True
    """Toggles advantages normalization"""


    expert_path:str="expert_policy.pth"
    """the weights file name for the expert policy. only work when there is no expert buffer"""
    discriminator_epochs:int=2
    """the update interval of the D/G"""
    policy_epochs:int=2
    """the update interval of the D/G"""


    batch_size:int=0
    """the batch size (computed in runtime)"""
    minibatch_size:int=0
    """the mini-batch size (computed in runtime)"""
    num_iterations:int=0
    """the number of iterations (computed in runtime)"""

def make_env(env_id,idx,capture_video,run_name)->Callable[[],gym.Env]:
    """

    :param env_id: the env_id of the gym environment
    :param idx: the index of the environment
    :param capture_video: whether to capture videos of the agent performances (check out 'videos' folder)
    :param run_name:
    :return: a function for env make
    """
    def thunk():
        if capture_video and idx==0:
            env=gym.make(env_id,render_mode="rgb_array")
            env=gym.wrappers.RecordVideo(env,f"videos/{run_name}")
        else:
            env=gym.make(env_id)
        env=gym.wrappers.RecordEpisodeStatistics(env,deque_size=1000)
        return env
    return thunk


def layer_init(layer,std=np.sqrt(2),bias_const=0.):
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self,envs):
        super().__init__()
        self.actor=nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,envs.single_action_space.n),std=0.01),
        )
        self.critic=nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
    def get_value(self,x):
        return self.critic(x).squeeze()
    def get_action_and_value(self,x,action=None):
        logits=self.actor(x)
        probs=Categorical(logits=logits)
        if action is None:
            action=probs.sample()
        return action,probs.log_prob(action),probs.entropy(),self.critic(x).squeeze()

class Discriminator(nn.Module):
    def __init__(self,envs):
        super().__init__()
        input_dim=np.array(envs.single_observation_space.shape).prod()+1
        self.discriminator=nn.Sequential(
            layer_init(nn.Linear(input_dim,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1)),
            nn.Sigmoid()
        )
    def forward(self,obs,action):
        action=action.unsqueeze(-1).float()
        x=torch.cat([obs,action],dim=-1)
        return self.discriminator(x)
    def calculate_reward(self,obs,action):
        B,N=obs.shape[0],obs.shape[1]
        obs=obs.view(B*N,-1)
        action=action.view(B*N)
        d=self(obs,action)
        d=d.view(B,N)
        return torch.log(d/(1-d))


class ReplayDataset(Dataset):
    def __init__(self,obs,actions,logprobs,expertobs,
                 expertactions):
        """

        """
        # 我们将前两个维度合并
        self.obs = obs.view(-1, *obs.shape[2:])  # 合并 num_steps 和 num_envs 维度
        self.actions = actions.flatten()
        self.logprobs = logprobs.flatten()
        perm=torch.randperm(expertactions.shape[0])
        self.expertobs = expertobs.view(-1, *obs.shape[2:])
        self.expertactions = expertactions.flatten()


    def __len__(self):
        return self.obs.size(0)

    def __getitem__(self, idx):
        """
        返回给定索引的数据。
        :param idx: 数据索引
        :return: 一个包含观察、动作和专家数据的元组
        """
        return (
            self.obs[idx],
            self.actions[idx],
            self.logprobs[idx],
            self.expertobs[idx],
            self.expertactions[idx]
        )


class PPOReplayDataset(Dataset):
    def __init__(self, obs, actions, values,logprobs,
                 advantages, returns):
        """

        """
        # 我们将前两个维度合并
        self.obs = obs.view(-1, *obs.shape[2:])  # 合并 num_steps 和 num_envs 维度
        self.actions = actions.flatten()
        self.values=values.flatten()
        self.logprobs = logprobs.flatten()
        self.advantages=advantages.flatten()
        self.returns=returns.flatten()


    def __len__(self):
        return self.obs.size(0)

    def __getitem__(self, idx):
        """
        返回给定索引的数据。
        :param idx: 数据索引
        :return: 一个包含观察、动作和专家数据的元组
        """
        return (
            self.obs[idx],
            self.actions[idx],
            self.values[idx],
            self.logprobs[idx],
            self.advantages[idx],
            self.returns[idx],
        )



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,

            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

    # seeding
    set_seed(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action is supported"

    agent = Agent(envs).to(device)
    policyoptimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    discriminator = Discriminator(envs).to(device)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate, eps=1e-5)

    # storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    expertactions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(
        device)

    length_list=[]

    # 构造 expert 模型后，加载预训练权重
    expert = Agent(envs).to(device)

    # 加载预训练模型的参数
    expert_ckpt_path = args.expert_path  # 你实际的模型路径
    expert.actor.load_state_dict(torch.load(expert_ckpt_path, map_location=device))
    for p in expert.parameters():
        p.requires_grad = False
    expert.eval()  # 设置为评估模式（关闭 dropout、batchnorm 等）

    # start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):

        if args.anneal_lr:
            frac = 1 - (iteration - 1.) / args.num_iterations
            lrnow = frac * args.learning_rate
            policyoptimizer.param_groups[0]["lr"] = lrnow
            discriminator_optimizer.param_groups[0]["lr"] = lrnow
        # get agent data
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            # get action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                expertaction,_,_,_=expert.get_action_and_value(next_obs)
            expertactions[step]=expertaction
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            dones[step] = next_done
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        length_list.append(info['episode']['r'])
                        print(f"global_step={global_step},episodic_return={info['episode']['r']}")
                        wandb.log({
                            "Charts/episodic_return": info["episode"]["r"],
                            "Charts/episodic_length": info["episode"]["l"],
                            "global_step": global_step
                        })

        if iteration % (args.discriminator_epochs + args.policy_epochs) <= args.discriminator_epochs:
            replay_dataset = ReplayDataset(
                obs=obs,
                actions=actions,
                logprobs=logprobs,
                expertobs=obs,
                expertactions=expertactions,
            )
            dataloader = DataLoader(replay_dataset, batch_size=args.minibatch_size, shuffle=True)

            for batch in dataloader:
                obs_batch, actions_batch, logprobs_batch, expertobs_batch, expertactions_batch = batch

                real_labels = torch.ones(obs_batch.size(0), 1).to(device)
                fake_labels = torch.zeros(obs_batch.size(0), 1).to(device)

                real_pred = discriminator(expertobs_batch, expertactions_batch)
                real_loss = F.binary_cross_entropy(real_pred, real_labels)

                fake_pred = discriminator(obs_batch, actions_batch)
                fake_loss = F.binary_cross_entropy(fake_pred, fake_labels)

                discriminator_loss = real_loss + fake_loss

                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)

                discriminator_optimizer.step()
            wandb.log({
                "Charts/discriminator_loss": discriminator_loss.item(),
                "global_step": global_step
            })
        else:
            # get advantage and returns
            with torch.no_grad():
                lastgaelam = 0
                rewards = discriminator.calculate_reward(obs, actions)
                next_value = agent.get_value(next_obs)
                advantages = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1 - next_done
                        next_value = next_value
                    else:
                        nextnonterminal = 1 - dones[t + 1]
                        next_value = values[t + 1]
                    delta = rewards[t] + args.gamma * next_value * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            replay_dataset = PPOReplayDataset(
                obs=obs,
                actions=actions,
                values=values,
                logprobs=logprobs,
                advantages=advantages,
                returns=returns,
            )
            dataloader = DataLoader(replay_dataset, batch_size=args.minibatch_size, shuffle=True)

            for batch in dataloader:
                obs_batch, actions_batch, oldvalues_batch, oldlogprobs_batch, advantages_batch, returns_batch = batch
                if args.norm_adv:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                for epoch in range(args.update_epochs):
                    # 使用判别器输出的概率计算损失
                    _, logprobs_batch, entropy_batch, values_batch = agent.get_action_and_value(obs_batch,
                                                                                                actions_batch)
                    logratio = logprobs_batch - oldlogprobs_batch
                    ratio = logratio.exp()

                    pg_loss1 = -advantages_batch * ratio
                    pg_loss2 = -advantages_batch * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    values_batch = values_batch.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (values_batch - returns_batch) ** 2
                        v_clipped = oldvalues_batch + torch.clamp(
                            values_batch - oldvalues_batch,
                            -args.clip_coef,
                            args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - returns_batch) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((values_batch - returns_batch) ** 2).mean()

                    entropy_loss = entropy_batch.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    policyoptimizer.zero_grad()
                    # 对策略网络进行反向传播
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    policyoptimizer.step()

            wandb.log({
                "Charts/policy_loss": loss.item(),
                "global_step": global_step
            })

    wandb.log({"average_length":np.mean(length_list).item()})


if __name__ =="__main__":
   main()