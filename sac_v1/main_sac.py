import pybullet_envs
import gym
import numpy as np

from actor_critic import Agent

# from utils import plot_learning_curve
from gym import wrappers
import wandb
from tqdm import tqdm

if __name__ == "__main__":

    config = dict(
        n_games=500,
        # env_name="InvertedPendulumBulletEnv-v0",
        env_name="AntBulletEnv-v0",
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

    with wandb.init(
        project="trashbot-sac",
        tags=[config["env_name"], "v1"],
        config=config,
        monitor_gym=True,
    ):
        config = wandb.config
        env = gym.make(config.env_name)
        agent = Agent(
            input_dims=env.observation_space.shape,
            env=env,
            n_actions=env.action_space.shape[0],
            # *config
        )

        env = gym.wrappers.Monitor(env, "tmp/video", force=True)

        wandb.watch(
            [
                agent.actor,
                agent.critic_1,
                agent.critic_2,
                agent.value,
                agent.target_value,
            ],
            log="all",
            log_freq=10,
        )

        best_score = env.reward_range[0]
        score_history = []
        load_checkpoint = False

        if load_checkpoint:
            agent.load_models()
            env.render(mode="human")

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
                    critic_loss, actor_loss, value_loss = agent.learn()

                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()

            print("episode ", i, "score %.1f" % score, "avg_score %.1f" % avg_score)
            print(f"actor_loss: {actor_loss} \t critic_loss: {critic_loss}")
            wandb.log(
                {
                    "score": score,
                    "avg_score": avg_score,
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                    "value_loss": value_loss,
                }
            )

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

