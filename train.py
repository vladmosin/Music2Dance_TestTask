from Agents import Actor, Critic
from Config import Config
from MemoryReplay import MemoryReplay
import sys
import numpy as np
import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm

from Noiser import Noiser
from Painter import draw


def td3_update(step):
    state, action, reward, next_state, done = replay_buffer.sample()
    critic1, critic2 = critic

    target_actions = noise.apply(actor.target(next_state), clip=True)
    expected_qvalue = torch.min(
        critic1.target(next_state, target_actions),
        critic2.target(next_state, target_actions),
    )
    expected_qvalue = reward.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * config.gamma * expected_qvalue
    expected_qvalue = expected_qvalue.detach()
    qvalue1 = critic1(state, action)
    qvalue2 = critic2(state, action)

    loss1 = F.mse_loss(qvalue1, expected_qvalue)
    update_net(critic1, loss1)

    loss2 = F.mse_loss(qvalue2, expected_qvalue)
    update_net(critic2, loss2)

    if step % config.td3_policy_delay == 0:
        actor_loss = - critic1(state, actor(state)).mean()
        update_net(actor, actor_loss)

        soft_update(critic1)
        soft_update(critic2)
        soft_update(actor)


def update_net(net, loss):
    net.optimizer.zero_grad()
    loss.backward()
    net.optimizer.step()


def soft_update(net):
    for target_param, param in zip(net.target.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - config.tau) + param.data * config.tau)


def test_episode():
    state = env.reset()
    done, episode_reward = False, 0
    while not done:
        action = actor.target.np_action(state)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward


def play_episode():
    state = env.reset()
    noise.update_eps()
    ep_reward = 0

    for step in range(config.max_timestamps):
        action = noise.apply(actor.np_action(state))
        next_state, reward, done, _ = env.step(action)

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        ep_reward += reward

        if len(replay_buffer) > config.batch_size:
            td3_update(step)

        if done:
            break

    return ep_reward


def test():
    return [test_episode() for _ in range(config.test_episodes)]


def train_draft():
    train_rewards = []
    test_rewards = []

    for id_episode in tqdm(range(1, config.max_episodes + 1), total=config.max_episodes):
        reward = play_episode()
        train_rewards.append(reward)

        if id_episode % config.test_interval == 0:
            test_rewards.append(
                np.mean(test())
            )
            print(f"episode: {id_episode} --- reward: {test_rewards[-1]:.3f}")

    return train_rewards, test_rewards


if __name__ == "__main__":
    config = Config.create_config(sys.argv[1:])
    env = gym.make(config.env_id)
    noise = Noiser(env.action_space, config)
    actor = Actor.define_actor(env, config)
    critic = Critic.define_critic(env, config)
    replay_buffer = MemoryReplay(config)
    train_rewards, test_rewards = train_draft()
    draw(train_rewards=train_rewards, test_rewards=test_rewards, config=config)