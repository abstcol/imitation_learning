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
    track:bool=False
    """if toggled, this experiment will be tracked with w&b"""
    wandb_project_name:str="imitation_learning"
    """the wandb's project name"""
    wandb_entity:str=None
    """the entity(team) of wandb's project"""
    capture_video:bool=False
    """whether to capture videos of the agent performances (check out 'videos' folder)"""

    env_id:str="CartPole-v1"
    """the if of the environment"""
    total_timesteps:int =500000
    """total timesteps of the experiments"""
    learning_rate:float=2.5e-4
    """the learning rate of the optimizer"""
    num_envs:int=4
    """the number of parallel game environments"""
    num_steps:int=128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr:bool=False
    """Toggle learning rate annealing for policy and value networks"""
    num_minibatches:int =4
    """the number of mini-batches"""
    update_epochs:int =1
    """the K epochs to update the policy"""
    ent_coef:float=0.01
    """coefficient of the entropy"""
    max_grad_norm:float=0.5

    expert_path:str="expert_policy.pth"
    """the weights file name for the expert policy"""
    discriminator_epochs:int=1
    """"""
    policy_epochs:int=1
    """"""

    batch_size:int=0
    """the batch size (computed in runtime)"""
    minibatch_size:int=0
    """the mini-batch size (computed in runtime)"""
    num_iterations:int=0
    """the number of iterations (computed in runtime)"""

def make_env(env_id,idx,capture_video,run_name)->Callable[[],gym.Env]:
    """

    :param env_id:
    :param idx:
    :param capture_video:
    :param run_name:
    :return:
    """
    def thunk():
        if capture_video and idx==0:
            env=gym.make(env_id,render_mode="rgb_array")
            env=gym.wrappers.RecordVideo(env,f"videos/{run_name}")
        else:
            env=gym.make(env_id)
        env=gym.wrappers.RecordEpisodeStatistics(env)
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
    def get_action(self,x,action=None):
        logits=self.actor(x)
        probs=Categorical(logits=logits)
        if action is None:
            action=probs.sample()
        return action,probs.log_prob(action),probs.entropy()

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


class ReplayDataset(Dataset):
    def __init__(self,obs,actions,logprobs,expertobs,expertactions,expertlogprobs):
        """
        初始化数据集。
        :param obs: 代理的观察数据 (tensor)
        :param actions: 代理的动作数据 (tensor)
        :param logprobs: 代理的 log 概率数据 (tensor)
        :param expertobs: 专家的观察数据 (tensor)
        :param expertactions: 专家的动作数据 (tensor)
        :param expertlogprobs: 专家的 log 概率数据 (tensor)
        """
        # 我们将前两个维度合并
        self.obs = obs.view(-1, *obs.shape[2:])  # 合并 num_steps 和 num_envs 维度
        self.actions = actions.flatten()
        self.logprobs = logprobs.flatten()
        self.expertobs = expertobs.view(-1, *expertobs.shape[2:])
        self.expertactions = expertactions.flatten()
        self.expertlogprobs = expertlogprobs.flatten()

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
            self.expertactions[idx],
            self.expertlogprobs[idx]
        )

if __name__ =="__main__":
    args=tyro.cli(Args)
    args.batch_size=int(args.num_envs*args.num_steps)
    args.minibatch_size=int(args.batch_size//args.num_minibatches)
    args.num_iterations=args.total_timesteps//args.batch_size
    run_name=f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )
    writer=SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"%("\n".join([f"|{key}|{value}|" for key,value in vars(args).items() ]))
    )

    #seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=args.torch_deterministic
    device=torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs=gym.vector.SyncVectorEnv(
        [make_env(args.env_id,i,args.capture_video,run_name) for i in range(args.num_envs)],
    )
    expertenvs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, run_name + "_expert") for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space,gym.spaces.Discrete),"only discrete action is supported"


    #env setup
    # 构造 expert 模型后，加载预训练权重
    expert = Agent(expertenvs).to(device)

    # 加载预训练模型的参数
    expert_ckpt_path = args.expert_path  # 你实际的模型路径
    expert.actor.load_state_dict(torch.load(expert_ckpt_path, map_location=device))
    for p in expert.parameters():
        p.requires_grad = False
    expert.eval()  # 设置为评估模式（关闭 dropout、batchnorm 等）

    agent=Agent(envs).to(device)
    policyoptimizer=optim.Adam(agent.parameters(),lr=args.learning_rate,eps=1e-5)

    discriminator=Discriminator(envs).to(device)
    discriminator_optimizer=optim.Adam(discriminator.parameters(),lr=args.learning_rate,eps=1e-5)

    #storage setup
    obs=torch.zeros((args.num_steps,args.num_envs)+envs.single_observation_space.shape).to(device)
    actions=torch.zeros((args.num_steps,args.num_envs)+envs.single_action_space.shape).to(device)
    logprobs=torch.zeros((args.num_steps,args.num_envs)).to(device)
    expertobs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    expertactions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    expertlogprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    #start the game
    global_step=0
    start_time=time.time()
    next_obs,_=envs.reset(seed=args.seed)
    next_obs=torch.Tensor(next_obs).to(device)
    next_done=torch.zeros(args.num_envs).to(device)

    expertnext_obs, _ = expertenvs.reset(seed=args.seed)
    expertnext_obs = torch.Tensor(expertnext_obs).to(device)
    expertnext_done = torch.zeros(args.num_envs).to(device)
    for iteration in range(1,args.num_iterations+1):

        if args.anneal_lr:
            frac=1-(iteration-1.)/args.num_iterations
            lrnow=frac*args.learning_rate
            policyoptimizer.param_groups[0]["lr"]=lrnow
            discriminator_optimizer.param_groups[0]["lr"]=lrnow
        #get agent data
        for step in range(args.num_steps):
            global_step+=args.num_envs
            obs[step]=next_obs

            #get action
            with torch.no_grad():
                action,logprob,_=agent.get_action(next_obs)

            actions[step]=action
            logprobs[step]=logprob

            next_obs,reward,terminations,truncations,infos=envs.step(action.cpu().numpy())
            next_done=np.logical_or(terminations,truncations)
            next_obs,next_done=torch.Tensor(next_obs).to(device),torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step},episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return",info["episode"]["r"],global_step)
                        writer.add_scalar("charts/episodic_length",info["episode"]["l"],global_step)




        #get expert data
        for step in range(args.num_steps):

            expertobs[step]=expertnext_obs

            #get action
            with torch.no_grad():
                expertaction,expertlogprob,_=expert.get_action(expertnext_obs)

            expertactions[step]=expertaction
            expertlogprobs[step]=expertlogprob

            expertnext_obs,expertreward,expertterminations,experttruncations,expertinfo=expertenvs.step(expertaction.cpu().numpy())
            expertnext_done=np.logical_or(expertterminations,experttruncations)
            expertnext_obs,expertnext_done=torch.Tensor(expertnext_obs).to(device),torch.Tensor(expertnext_done).to(device)

        replay_dataset=ReplayDataset(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            expertobs=expertobs,
            expertactions=expertactions,
            expertlogprobs=expertlogprobs
        )
        dataloader=DataLoader(replay_dataset,batch_size=args.minibatch_size,shuffle=True)


        if iteration%(args.discriminator_epochs+args.policy_epochs)<args.discriminator_epochs:

            for epoch in range(args.update_epochs):
                for batch in dataloader:

                    obs_batch,actions_batch,logprobs_batch,expertobs_batch,expertactions_batch,expertlogprobs_batch=batch

                    real_labels=torch.ones(obs_batch.size(0),1).to(device)
                    fake_labels=torch.zeros(obs_batch.size(0),1).to(device)

                    real_pred=discriminator(expertobs_batch,expertactions_batch)
                    real_loss=F.binary_cross_entropy(real_pred,real_labels)

                    fake_pred=discriminator(obs_batch,actions_batch)
                    fake_loss=F.binary_cross_entropy(fake_pred,fake_labels)

                    entropy_loss=-torch.mean(logprobs_batch*torch.exp(logprobs_batch))

                    discriminator_loss=real_loss+fake_loss+args.ent_coef*entropy_loss


                    discriminator_optimizer.zero_grad()
                    discriminator_loss.backward()
                    nn.utils.clip_grad_norm_(discriminator.parameters(),args.max_grad_norm)


                    discriminator_optimizer.step()

            writer.add_scalar("charts/discriminator_loss",discriminator_loss.item(),global_step)
        else:
            for epoch in range(args.update_epochs):
                for batch in dataloader:
                    obs_batch, actions_batch, logprobs_batch, expertobs_batch, expertactions_batch, expertlogprobs_batch = batch

                    real_labels = torch.ones(obs_batch.size(0), 1).to(device)
                    # 使用判别器输出的概率计算损失

                    fake_pred = discriminator(obs_batch, actions_batch)
                    fake_loss = F.binary_cross_entropy(fake_pred, real_labels)
                    entropy_loss=-torch.mean(logprobs_batch*torch.exp(logprobs_batch))
                    # 计算策略网络损失，目标是最大化 expert 数据的概率（最小化判别器的预测概率）
                    policy_loss = fake_loss+args.ent_coef*entropy_loss  # 取负号，因为我们希望最大化该值

                    policyoptimizer.zero_grad()
                    # 对策略网络进行反向传播
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(),args.max_grad_norm)

                    policyoptimizer.step()
            writer.add_scalar("charts/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("charts/global_step",global_step,global_step)