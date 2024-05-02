import os
from datetime import datetime
import gym
import json
from reinforcement_learning.agent.dqn_agent import DQNAgent
from reinforcement_learning.train_cartpole import run_episode
from reinforcement_learning.agent.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v0").unwrapped

    device = torch.device("cpu")

    state_dim = 4
    num_actions = 2

    q_net = MLP(state_dim, num_actions, 400)
    t_net = MLP(state_dim, num_actions, 400)
    dqn = DQNAgent(q_net, t_net, num_actions, device=device)

    model_name = "dqn_agent200"
    dqn.load(f"models_cartpole/{model_name}.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, dqn, deterministic=True, do_training=False, rendering=True
        )
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["model_name"] = model_name
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print("... finished")
