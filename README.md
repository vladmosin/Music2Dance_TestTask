## Implementation of TD3 algorithm

Constants are set in config to work on "BipedalWalker-v2" environment

To run program with default settings just type `python train.py`

The next settings may be changed:
gamma, tau, actor_lr, critic_lr,
max_episodes, max_timestamps,
sigma, theta,
eps_min, eps_max,
capacity, td3_policy_delay,
test_episodes, env_id,
batch_size

To change parameter run python train.py --parameter value

Graphics are saved in "graphics" folder 
