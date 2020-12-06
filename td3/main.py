import pybullet_envs
import gym
import numpy as np
import torch as T

from actor_critic import Agent
from gym import wrappers
import wandb
from tqdm import tqdm


def main(config):
    env = gym.make(config["env_name"])
    # record videos of the agent playing the game
    env = gym.wrappers.Monitor(
        env,
        "tmp/video",
        video_callable=(lambda episode_id: episode_id % 10 == 0),
        force=True,
    )

    with wandb.init(
        project="trashbot-sac",
        tags=[config["env_name"], "td3"],
        config=config,
        monitor_gym=True,
    ):
        config = wandb.config

        agent = Agent(
            env=env,
            input_dims=env.observation_space.shape,
            n_actions=env.action_space.shape[0],
            gamma=config["gamma"],
            max_size=config["max_size"],
            tau=config["tau"],
            lr=config["lr"],
            layer1_size=config["layer1_size"],
            layer2_size=config["layer2_size"],
            batch_size=config["batch_size"],
        )

        wandb.watch(
            [
                agent.actor,
                agent.critic_1,
                agent.critic_2,
                agent.target_actor,
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
                    (loss_p, loss_q, target_actions) = agent.learn()
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
            print(f"episode {i} score {score} avg_score {avg_score}")
            print("loss_p, loss_q, target_actions")
            print(loss_p, loss_q, target_actions)
            wandb.log(
                {
                    "score": score,
                    "avg_score": avg_score,
                    "loss_p": loss_p,
                    "loss_q": loss_q,
                    "target_actions": target_actions,
                }
            )

        # WHEN DONE TRIANING TODO: ake it save periodically
        # Save the model in the exchangeable ONNX format
        print("SAVING MODELS AFTER TRAINING..")
        print("OBSERVATION", observation)
        print("\nACTION", action)
        device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        state, action, reward, new_state, done = agent.memory.sample_buffer(
            config.batch_size
        )

        T.onnx.export(
            agent.actor,
            T.tensor(observation, dtype=T.float, device=device),
            "actor.onnx",
        )

        T.onnx.export(
            agent.critic_1,
            (
                T.tensor(state, dtype=T.float, device=device),
                T.tensor(action, dtype=T.float, device=device),
            ),
            "critic_1.onnx",
        )

        T.onnx.export(
            agent.critic_2,
            (
                T.tensor(state, dtype=T.float, device=device),
                T.tensor(action, dtype=T.float, device=device),
            ),
            "critic_2.onnx",
        )

        T.onnx.export(
            agent.target_actor,
            T.tensor(observation, dtype=T.float, device=device),
            "target_actor.onnx",
        )

        T.onnx.export(
            agent.target_critic_1,
            (
                T.tensor(state, dtype=T.float, device=device),
                T.tensor(action, dtype=T.float, device=device),
            ),
            "target_critic_1.onnx",
        )

        T.onnx.export(
            agent.target_critic_2,
            (
                T.tensor(state, dtype=T.float, device=device),
                T.tensor(action, dtype=T.float, device=device),
            ),
            "target_critic_2.onnx",
        )

        wandb.save("actor.onnx")
        wandb.save("critic_1.onnx")
        wandb.save("critic_2.onnx")
        wandb.save("target_critic_1.onnx")
        wandb.save("target_critic_2.onnx")

    # # TEST MODE
    # if load_checkpoint:
    #     agent.load_models()
    #     env.render(mode="human")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--n_games", type=int, default=1000)
    args = parser.parse_args()

    config = dict(
        n_games=args.n_games,
        # env_name="InvertedPendulumBulletEnv-v0",
        # env_name="AntBulletEnv-v0",
        # env_name="LunarLanderContinuous-v2",
        env_name=args.env,
        gamma=0.99,
        max_size=1_000_000,
        tau=0.005,
        lr=0.001,
        layer1_size=400,
        layer2_size=300,
        batch_size=100,
    )
    main(config)
