from __future__ import print_function

import gym
from reinforcement_learning.agent.dqn_agent import DQNAgent
from reinforcement_learning.train_carracing import run_episode
from reinforcement_learning.agent.networks import *
import numpy as np
import json
import os
from datetime import datetime

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length = 5
    device = torch.device("mps")
    print(device)
    num_classes = 5

    q_net = CNN(history_length=history_length, n_classes=num_classes).to(device)
    t_net = CNN(history_length=history_length, n_classes=num_classes).to(device)
    dqn = DQNAgent(q_net, t_net, num_classes, device)

    dqn.load("models_carracing/cr_dqn_agent_m2_p2.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        print(f"Episode {i}")
        stats = run_episode(
            env,
            dqn,
            deterministic=False,
            do_training=False,
            rendering=True,
            history_length=history_length,
            skip_frames=5,
            acceleration=0.2,
        )
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/carracing_results_0_dqn-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print("... finished")
