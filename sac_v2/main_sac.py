import pybullet_envs
import gym
import numpy as np
import torch

from actor_critic import Agent
from gym import wrappers
import wandb
from tqdm import tqdm

# from tqdm.notebook import tqdm

# from utils import plot_learning_curve

if __name__ == "__main__":
    config = dict(
        n_games=250,
        env_name="InvertedPendulumBulletEnv-v0",
        alpha=0.2,
        gamma=0.99,
        max_size=1_000_000,
        tau=0.005,
        lr=1e-3,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
    )

    with wandb.init(project="trashbot-sac", config=config):
        config = wandb.config
        env = gym.make(config.env_name)
        agent = Agent(
            input_dims=env.observation_space.shape,
            env=env,
            n_actions=env.action_space.shape[0],
        )

        # env = wrappers.Monitor(
        #     env, "tmp/video", video_callable=lambda episode_id: True, force=True
        # )

        wandb.watch(
            [
                agent.actor,
                agent.critic_1,
                agent.critic_2,
                agent.target_critic_1,
                agent.target_critic_2,
            ],
            log="all",
            log_freq=10,
        )

        best_score = env.reward_range[0]
        score_history = []
        load_checkpoint = False

        for i in tqdm(range(config.n_games)):

            observation = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                if not load_checkpoint:
                    loss_q, loss_q1, loss_q2, loss_p = agent.learn()
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                # if not load_checkpoint:
                #     agent.save_model()
            print(f"episode {i} score {score} avg_score {avg_score}")
            wandb.log(
                {
                    "episode": i,
                    "score": score,
                    "avg_score": avg_score,
                    "loss_q": loss_q,
                    "loss_q1": loss_q1,
                    "loss_q2": loss_q2,
                    "loss_p": loss_p,
                }
            )

            # Save the model in the exchangeable ONNX format
            torch.onnx.export(
                [agent.actor, agent.critic_2, agent.critic_2], observation, "model.onnx"
            )
            wandb.save("model.onnx")

    # TEST MODE
    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    # filename = "inverted_pendulum.png"
    # figure_file = "plots/" + filename
    # if not load_checkpoint:
    #     x = [i + 1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)
