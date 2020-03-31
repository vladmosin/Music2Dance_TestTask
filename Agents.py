from torch import nn
import torch
import copy
from torch import optim
from Config import Config


class Actor(nn.Module):
    def __init__(self, obs_size, act_size, config: Config):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh(),
        )
        self.device = config.device

    def forward(self, x):
        return self.net(x)

    def np_action(self, state):
        action = self.forward(torch.tensor(state, dtype=torch.float, device=self.device))
        return action.detach().cpu().numpy()

    @staticmethod
    def define_actor(env, config: Config):
        actor_local = Actor(env.observation_space.shape[0], env.action_space.shape[0], config).to(config.device)
        actor_target = copy.deepcopy(actor_local)
        actor_optimizer = optim.Adam(actor_local.parameters(), lr=config.actor_lr)
        actor = actor_local
        actor.target = actor_target
        actor.optimizer = actor_optimizer

        return actor


class Critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Critic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, action):
        obs = self.obs_net(state)
        return self.out_net(torch.cat([obs, action], dim=1))

    @staticmethod
    def get_critic(env, config: Config):
        critic_local = Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(config.device)
        critic_target = copy.deepcopy(critic_local)
        critic_optimizer = optim.Adam(critic_local.parameters(), lr=config.critic_lr)
        critic = critic_local
        critic.target = critic_target
        critic.optimizer = critic_optimizer

        return critic

    @staticmethod
    def define_critic(env, config: Config):
        return Critic.get_critic(env, config), Critic.get_critic(env, config)
