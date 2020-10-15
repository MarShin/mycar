import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

# TODO: variable input_dims & n_actions
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
        tau=0.004,
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

        self.critic_1 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_2"
        )
        self.target_critic_1 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="target_critic_1"
        )
        self.targert_critic_2 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="targert_critic_2"
        )

        self.scale = reward_scale
        # hard update target network when fist init, afters soft update by tau
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_c1_params = self.target_critic_1.named_parameters()
        target_c2_params = self.targert_critic_2.named_parameters()
        c1_params = self.critic_1.named_parameters()
        c2_params = self.critic_2.named_parameters()

        target_c1_state_dict = dict(target_c1_params)
        target_c2_state_dict = dict(target_c2_params)
        c1_state_dict = dict(c1_params)
        c2_state_dict = dict(c2_params)

        # TODO: spinningup here use .copy_ of tensor obj

