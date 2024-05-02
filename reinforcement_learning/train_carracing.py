# export DISPLAY=:0

import sys

sys.path.append("../")

import numpy as np
import torch
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray
from utils import *
from reinforcement_learning.agent.dqn_agent import DQNAgent
from reinforcement_learning.agent.networks import *
import pprint


def run_episode(
    env,
    agent,
    deterministic,
    skip_frames=0,
    do_training=True,
    rendering=False,
    max_timesteps=1000,
    history_length=0,
    acceleration=0.8,
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    state = torch.tensor(state).permute(2, 0, 1)

    while True:
        state = torch.tensor(state)
        action_id = agent.act(
            state=state,
            deterministic=deterministic,
            random_probability=[0.3, 0.2, 0.2, 0.2, 0.1],
        )
        action = id_to_action(action_id, acceleration)
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    history_length=0,
    model_dir="./models_carracing",
    tensorboard_dir="./tensorboard",
):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "train"),
        "Carracing",
        stats=["episode_reward", "straight", "left", "right", "accel", "brake"],
    )
    tensorboard_ev = Evaluation(
        os.path.join(tensorboard_dir, "train"),
        "Carracing",
        stats=["evaluation_episodes", "average_reward"],
    )

    for i in range(num_episodes):
        print("epsiode %d" % i)

        max_timesteps = min(((i // 5) + 1) * 100, 2000)
        print("max timesteps: %d" % max_timesteps)

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            do_training=True,
            history_length=history_length,
            rendering=False,
            skip_frames=5,
            acceleration=0.5,
        )
        eval_dict_ = {
            "episode_reward": stats.episode_reward,
            "straight": stats.get_action_usage(STRAIGHT),
            "left": stats.get_action_usage(LEFT),
            "right": stats.get_action_usage(RIGHT),
            "accel": stats.get_action_usage(ACCELERATE),
            "brake": stats.get_action_usage(BRAKE),
        }

        tensorboard.write_episode_data(
            i,
            eval_dict=eval_dict_,
        )

        if i % eval_cycle == 0 or i == num_episodes - 1:
            total_reward = 0
            for j in range(num_eval_episodes):
                eval_stats = run_episode(
                    env,
                    agent,
                    deterministic=True,
                    do_training=False,
                    history_length=history_length,
                    acceleration=0.5,
                )
                total_reward += eval_stats.episode_reward
            avg_reward = total_reward / num_eval_episodes
            print(
                "Average reward over",
                num_eval_episodes,
                "evaluation episodes:",
                avg_reward,
            )
            # Log average reward to tensorboard or any other logging mechanism
            tensorboard_ev.write_episode_data(
                i,
                eval_dict={
                    "evaluation_episodes": num_eval_episodes,
                    "average_reward": avg_reward,
                },
            )

        # store model.
        if i % eval_cycle == 0 or i == num_episodes - 1:
            agent.save(os.path.join(model_dir, "carracing_agent.pt"))

    tensorboard.close_session()
    tensorboard_ev.close_session()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20

    history_length = 5
    num_classes = 5

    device = torch.device("mps")
    print(device)

    env = gym.make("CarRacing-v0").unwrapped

    q_net = CNN(history_length=history_length, n_classes=num_classes).to(device)
    t_net = CNN(history_length=history_length, n_classes=num_classes).to(device)
    dqn = DQNAgent(q_net, t_net, num_classes, device)

    # dqn.load('models_carracing/cr_dqn_agent_m2_p2.pt')

    train_online(
        env,
        dqn,
        num_episodes=200,
        history_length=history_length,
        model_dir="./models_carracing",
    )
