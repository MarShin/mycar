import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(
        self, lr, input_dims, act_dims, fc1_dims=256, fc2_dims=256, name="critic",
    ):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.act_dims = act_dims
        self.name = name

        # TODO: just the [0] dim of state is good enough?
        self.fc1 = nn.Linear(self.input_dims[0] + act_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return T.squeeze(q, -1)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ActorNetwork(nn.Module):
    def __init__(
        self,
        lr,
        input_dims,
        act_dims,
        act_limit,
        fc1_dims=256,
        fc2_dims=256,
        name="actor",
    ):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.act_dims = act_dims
        self.name = name
        self.act_limit = act_limit
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.act_dims)
        # self.sigma = nn.Linear(self.fc2_dims, self.act_dims)
        self.log_std = nn.Linear(self.fc2_dims, self.act_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        log_std = self.log_std(prob)
        log_std = T.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        sigma = T.exp(log_std)

        # clamp sigma coz can't have infite wide distribution
        # TODO: see how spinningup is clamping - normal the linear multiply noise?
        # instead of [-20, +2] as used in paper
        # sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True, deterministic=False):
        mu, sigma = self.forward(state)
        pi_dist = Normal(mu, sigma)

        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_dist.rsample()

        # action here is scaled to the env max action space as Tanh outputs [-1,+1]

        # for more details check paper v2 appendix C
        # SpinningUp & RLKit have a more numerical stable approach
        # Adapted from
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        # This formula is mathematically equivalent to log(1 - tanh(x)^2).

        # Derivation:
        # log(1 - tanh(x)^2)
        #  = log(sech(x)^2)
        #  = 2 * log(sech(x))
        #  = 2 * log(2e^-x / (e^-2x + 1))
        #  = 2 * (log(2) - x - log(e^-2x + 1))
        #  = 2 * (log(2) - x - softplus(-2x))
        if reparameterize:
            logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        # action = T.tanh(actions) * T.tensor(self.act_limit).to(self.device)
        # log_probs = pi_dist.log_prob(actions)
        # log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        # log_probs = log_probs.sum(1, keepdim=True)

        pi_action = T.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi
