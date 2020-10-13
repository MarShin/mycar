import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork


class Agent:
    def __init__(
        self,
        alpha=0.0003,
        beta=0.0003,
        input_dims=[8],
        env=None,
        gamma=0.99,
        n_actions=2,
        max_size=1_000_000,
        tau=0.005,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
    ):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(
            alpha,
            input_dims,
            n_actions=n_actions,
            name="actor",
            max_action=env.action_space.high,
        )

        self.critic1 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_1"
        )
        self.critic2 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_2"
        )
        self.value = ValueNetwork(beta, input_dims, name="value")
        self.target_value = ValueNetwork(beta, input_dims, name="target_value")

        self.scale = reward_scale
        # hard update target network when fist init, afters soft update by tau
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        # pytorch magic syntax
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()


# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
