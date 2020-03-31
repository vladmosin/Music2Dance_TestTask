from argparse import ArgumentParser
import torch


class Config:
    def __init__(self, gamma, tau,
                 actor_lr, critic_lr,
                 max_episodes, max_timestamps,
                 sigma, theta,
                 eps_min, eps_max,
                 capacity, td3_policy_delay,
                 test_episodes, env_id,
                 batch_size):
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_episodes = max_episodes
        self.max_timestamps = max_timestamps
        self.sigma = sigma
        self.theta = theta
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.capacity = capacity
        self.env_id = env_id
        self.td3_policy_delay = td3_policy_delay
        self.test_episodes = test_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_interval = self.max_episodes // 50 if self.max_episodes >= 50 else 1

    @staticmethod
    def create_config(commandline_args):
        parser = ArgumentParser()
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--tau", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--actor_lr", type=float, default=2e-4)
        parser.add_argument("--critic_lr", type=float, default=1e-4)
        parser.add_argument("--max_episodes", type=int, default=1000)
        parser.add_argument("--max_timestamps", type=int, default=500)
        parser.add_argument("--sigma", type=float, default=0.6)
        parser.add_argument("--theta", type=float, default=0.15)
        parser.add_argument("--eps_min", type=float, default=0.2)
        parser.add_argument("--eps_max", type=float, default=1)
        parser.add_argument("--capacity", type=int, default=50000)
        parser.add_argument("--test_episodes", type=int, default=3)
        parser.add_argument("--td3_policy_delay", type=int, default=2)
        parser.add_argument("--env_id", default="BipedalWalker-v2")

        args = parser.parse_args(commandline_args)

        return Config(gamma=args.gamma,
                      tau=args.tau,
                      actor_lr=args.actor_lr,
                      critic_lr=args.critic_lr,
                      max_episodes=args.max_episodes,
                      max_timestamps=args.max_timestamps,
                      sigma=args.sigma,
                      theta=args.theta,
                      eps_min=args.eps_min,
                      eps_max=args.eps_max,
                      capacity=args.capacity,
                      td3_policy_delay=args.td3_policy_delay,
                      test_episodes=args.test_episodes,
                      env_id=args.env_id,
                      batch_size=args.batch_size
                      )
