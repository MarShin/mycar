import pybullet_envs
import gym
import numpy as np

from actor_critic import Agent
from utils import plot_learning_curve
from gym import wrappers

if __name__ == "__main__":
    env = gym.make("InvertedPendulumBulletEnv-v0")
    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
    )
    n_games = 250

    # record videos of teh agent playing the game
    env = wrappers.Monitor(
        env, "tmp/video", video_callable=lambda episode_id: True, force=True
    )
    filename = "inverted_pendulum.png"

    figure_file = "plots/" + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                critic_loss, actor_loss = agent.learn()
                # agent.learn()

            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print("episode ", i, "score %.1f" % score, "avg_score %.1f" % avg_score)
        print(f"actor_loss: {actor_loss} \t critic_loss: {critic_loss}")

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

