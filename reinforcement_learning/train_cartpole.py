import sys

sys.path.append("../")

import numpy as np
import gym
import itertools as it
from reinforcement_learning.agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from reinforcement_learning.agent.networks import *
from utils import EpisodeStats
import pprint


def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    eval_cycle,
    num_eval_episodes,
    model_dir="./models_cartpole",
    tensorboard_dir="./tensorboard",
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "train"), 
        "CartPole", 
        ["episode_reward", "a_0", "a_1"]
    )
    tensorboard_ev = Evaluation(
        os.path.join(tensorboard_dir, "train"),
        "Carracing",
        stats=["evaluation_episodes", "average_reward"]
    )

    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True, rendering=True)
        eval_dict_ = {
                "episode_reward": stats.episode_reward,
                "a_0": stats.get_action_usage(0),
                "a_1": stats.get_action_usage(1),
            }
        tensorboard.write_episode_data(
            i,
            eval_dict=eval_dict_,
        )

        if i % eval_cycle == 0 or i==num_episodes-1:
            total_reward = 0
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False)
                total_reward += eval_stats.episode_reward
            avg_reward = total_reward / num_eval_episodes
            print("Average reward over", num_eval_episodes, "evaluation episodes:", avg_reward)
            # Log average reward to tensorboard or any other logging mechanism
            tensorboard_ev.write_episode_data(
                i,
                eval_dict={
                    "evaluation_episodes": num_eval_episodes,
                    "average_reward": avg_reward
                }
            )


        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agentt_200.pt"))

    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 10  # evaluate every 10 episodes

    device = torch.device("cpu")

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    q_net = MLP(state_dim, num_actions, 400)
    t_net = MLP(state_dim, num_actions, 400)
    dqn = DQNAgent(q_net, t_net, num_actions, device=device)
    train_online(env, dqn, 200, eval_cycle, num_eval_episodes, model_dir="./models_final_cartpole")
