import os
import itertools
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        input_dims=[8],
        env=None,
        n_actions=2,
        alpha=0.2,
        gamma=0.99,
        max_size=1_000_000,
        tau=0.005,
        lr=1e-3,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
    ):
        self.alpha = alpha  # ideally auto learn temperature instead of fixed
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(
            lr,
            input_dims,
            n_actions=n_actions,
            name="actor",
            max_action=env.action_space.high,
        )

        self.critic_1 = CriticNetwork(
            lr, input_dims, n_actions=n_actions, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            lr, input_dims, n_actions=n_actions, name="critic_2"
        )
        self.target_critic_1 = CriticNetwork(
            lr, input_dims, n_actions=n_actions, name="target_critic_1"
        )
        self.target_critic_2 = CriticNetwork(
            lr, input_dims, n_actions=n_actions, name="target_critic_2"
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

    def soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        self.soft_update(source=self.critic_1, target=self.target_critic_1, tau=tau)
        self.soft_update(source=self.critic_2, target=self.target_critic_2, tau=tau)

    def save_model(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

    def compute_q_loss(self, reward, done, state, state_, action):

        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)
        #         print('q1 size', q1.size(), '\n')

        with T.no_grad():
            # target action from the CURRENT plicy
            action_, log_probs_ = self.actor.sample_normal(state, reparameterize=False)
            log_probs_ = log_probs_.view(-1)

            # target Q values
            q1_targ = self.target_critic_1(state_, action_).view(-1)
            q2_targ = self.target_critic_2(state_, action_).view(-1)
            q_targ = T.min(q1_targ, q2_targ)
            #             print('q_targ size', q_targ.size(), '\n')
            backup = reward + self.gamma * (1 - done) * (
                q_targ - self.alpha * log_probs_
            )
        #             print('log_probs_ size', log_probs_.size())
        #             print('backup size', backup.size(), '\n')

        #         broadcasting error
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)
        loss_q = loss_q1 + loss_q2

        return loss_q, loss_q1, loss_q2

    def compute_p_loss(self, reward, done, state, state_, action):
        action_, log_probs_ = self.actor.sample_normal(state, reparameterize=True)
        # print("LOG PROB SIZE: ", log_probs_.size())
        log_probs_ = log_probs_.view(-1)
        q1 = self.critic_1(state, action_)
        q2 = self.critic_2(state, action_)
        q = T.min(q1, q2)
        # print("min q value size: ", q.size())
        q = q.view(-1)

        # Entropy-regularized policy loss
        loss_p = (self.alpha * log_probs_ - q).mean()

        return loss_p, log_probs_, action_

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
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        loss_q, loss_q1, loss_q2 = self.compute_q_loss(
            reward, done, state, state_, action
        )
        loss_q.backward()
        # alternative in spinningup up concat params of q1 & q2 and Adam together
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # backprop actor network
        # TODO: freeze gradients of 2 Q networks when updating policy
        self.actor.optimizer.zero_grad()
        loss_p, log_probs_, action_ = self.compute_p_loss(
            reward, done, state, state_, action
        )
        loss_p.backward()
        self.actor.optimizer.step()

        # update target netowrks
        self.update_network_parameters()

        return (loss_q, loss_q1, loss_q2, loss_p, log_probs_, action_)

