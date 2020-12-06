import os
import itertools
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
import itertools


class Agent:
    def __init__(
        self,
        env,
        input_dims,
        act_dims,
        act_limit,
        gamma,
        max_size,
        tau,
        lr,
        alpha=0.2,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
    ):
        self.alpha = alpha  # ideally auto learn temperature instead of fixed
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, act_dims)
        self.batch_size = batch_size
        self.act_dims = act_dims

        self.actor = ActorNetwork(
            lr=lr,
            input_dims=input_dims,
            act_dims=act_dims,
            act_limit=act_limit,
            name="actor",
        )

        self.critic_1 = CriticNetwork(
            lr=lr, input_dims=input_dims, act_dims=act_dims, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            lr=lr, input_dims=input_dims, act_dims=act_dims, name="critic_2"
        )
        self.target_critic_1 = CriticNetwork(
            lr=lr, input_dims=input_dims, act_dims=act_dims, name="target_critic_1"
        )
        self.target_critic_2 = CriticNetwork(
            lr=lr, input_dims=input_dims, act_dims=act_dims, name="target_critic_2"
        )

        # hard update target network when fist init, afters soft update by tau
        self.update_network_parameters(tau=1)
        self.q_params = itertools.chain(
            self.critic_1.parameters(), self.critic_2.parameters()
        )
        self.q_optimizer = Adam(q_params, lr=lr)

    def choose_action(self, observation):
        with T.no_grad():
            state = T.Tensor([observation]).to(self.actor.device)
            actions, _ = self.actor.sample_normal(state, reparameterize=False)

            # pytorch magic syntax: .cpu().detach().numpy()[0]
            return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def soft_update(self, source, target, tau):
        with T.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        self.soft_update(source=self.critic_1, target=self.target_critic_1, tau=tau)
        self.soft_update(source=self.critic_2, target=self.target_critic_2, tau=tau)

    def compute_q_loss(self, reward, done, state, state_, action):

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        with T.no_grad():
            # target action from the CURRENT policy
            action_, log_probs_ = self.actor.sample_normal(state_, reparameterize=True)

            # target Q values
            q1_targ = self.target_critic_1(state_, action_)
            q2_targ = self.target_critic_2(state_, action_)
            q_targ = T.min(q1_targ, q2_targ)
            backup = reward + self.gamma * (1 - done) * (
                q_targ - self.alpha * log_probs_
            )

        loss_q1 = 0.5 * F.mse_loss(q1, backup)
        loss_q2 = 0.5 * F.mse_loss(q2, backup)
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(
            Q1Vals=q1.cpu().detach().numpy(), Q2Vals=q2.cpu().detach().numpy()
        )

        return loss_q, loss_q1, loss_q2, q_info

    def compute_p_loss(self, reward, done, state, state_, action):
        action_, log_probs_ = self.actor.sample_normal(state, reparameterize=True)
        # print("LOG PROB SIZE: ", log_probs_.size())
        q1 = self.critic_1(state, action_)
        q2 = self.critic_2(state, action_)
        q = T.min(q1, q2)
        # print("min q value size: ", q.size())

        # Entropy-regularized policy loss
        loss_pi = -T.mean(q - self.alpha * log_probs_)

        # Useful info for logging
        pi_info = dict(LogPi=log_probs_.cpu().detach().numpy())

        return loss_pi, pi_info

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None, None, None, None, None, None

        # sample from Replay Buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # backprop 2 critic networks
        # self.critic_1.optimizer.zero_grad()
        # self.critic_2.optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        loss_q, loss_q1, loss_q2, q_info = self.compute_q_loss(
            reward, done, state, state_, action
        )
        loss_q.backward()
        # alternative in spinningup up concat params of q1 & q2 and Adam together
        # self.critic_1.optimizer.step()
        # self.critic_2.optimizer.step()
        self.q_optimizer.step()

        # backprop actor network
        # TODO: freeze gradients of 2 Q networks when updating policy
        self.actor.optimizer.zero_grad()
        loss_p, pi_info = self.compute_p_loss(reward, done, state, state_, action)
        loss_p.backward()
        self.actor.optimizer.step()

        # update target netowrks
        self.update_network_parameters()

        return (
            q_info,
            pi_info,
            loss_q.detach(),
            loss_q1.detach(),
            loss_q2.detach(),
            loss_p.detach(),
        )

