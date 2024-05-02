import numpy as np
import torch
import torch.optim as optim
from reinforcement_learning.agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        device,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-4,
        history_length=0,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        self.device = device
        # setup networks
        self.Q = Q
        self.Q_target = Q_target
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        # Add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        # Sample mini-batch from the replay buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.next_batch(
            self.batch_size
        )

        try:
            states_batch = torch.stack(tuple(states), dim=0).to(device=self.device)
        except TypeError:
            states_batch = torch.tensor(tuple(states)).to(device=self.device)
        actions_batch = torch.tensor(
            actions.astype(int), dtype=torch.int64, device=self.device
        )
        next_states_batch = torch.tensor(
            next_states.astype(float), dtype=torch.float, device=self.device
        )
        rewards_batch = torch.tensor(
            rewards.astype(float), dtype=torch.float, device=self.device
        )
        terminal_batch = torch.tensor(
            dones.astype(float), dtype=torch.float, device=self.device
        )

        # Compute TD targets and loss
        Q_values_next = (
            self.Q_target(next_states_batch).max(dim=1)[0].detach()
        )  # Detach to avoid backpropagation
        terminal_numerical = terminal_batch.float()
        td_target = (
            rewards_batch + (1 - terminal_numerical) * self.gamma * Q_values_next
        )
        Q_values = self.Q(states_batch)
        td_estimate = Q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

        loss = self.loss_function(td_estimate, td_target)

        # Update the Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update for target network
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic, random_probability=[]):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """

        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # Take greedy action (argmax)
            Q_values = self.Q(
                torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                    0
                )
            )
            action_id = Q_values.max(dim=1)[1].item()
        else:
            # Sample random action
            if len(random_probability) == 0:
                action_id = np.random.choice(self.num_actions)
            else:
                action_id = np.random.choice(self.num_actions, p=random_probability)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
