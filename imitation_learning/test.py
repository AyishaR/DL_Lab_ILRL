import sys

sys.path.append(".")
from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from torch.nn import *

from imitation_learning.agent.bc_agent import BCAgent
import matplotlib.pyplot as plt
from utils import *


def state_preprocessing(state):
    state = rgb2gray(state)
    state[85:, :15] = 0.0  # hide reward in the image
    return state


def run_episode(env, agent, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0
    image_hist = []

    state = env.reset()

    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()

    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1).copy()
    state = torch.tensor(state).permute(2, 0, 1)

    steps = 0

    while True:
        state = torch.tensor(state).unsqueeze(0)

        steps += 1

        if steps < 40:
            # Accelerate initially - to get the car started
            output = 3
        else:
            _, output = agent.predict(state)
            output = output.detach().numpy()

        action = id_to_action(output, max_speed=0.1)

        next_state, r, done, info = env.step(action)
        episode_reward += r

        step += 1

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist)

        state = next_state.copy()

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    # history_length = 5

    n_test_episodes = 15  # number of episodes to test

    for history_length in [5]:
        # TODO: load agent
        agent = BCAgent(history_length, n_classes=5)
        mname = f"agent_data5_h{history_length}_v18"
        agent.load(f"models_im/{mname}.pt")

        env = gym.make("CarRacing-v0").unwrapped

        episode_rewards = []
        for i in range(n_test_episodes):
            episode_reward = run_episode(
                env, agent, rendering=rendering, max_timesteps=2000
            )
            episode_rewards.append(episode_reward)

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["model"] = mname
        results["episode_rewards"] = episode_rewards
        results["mean"] = np.array(episode_rewards).mean()
        results["std"] = np.array(episode_rewards).std()

        fname = (
            f"results/results_bc_agent-{history_length}"
            + "-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        fh = open(fname, "w")
        json.dump(results, fh)

        env.close()
        print("... finished")
