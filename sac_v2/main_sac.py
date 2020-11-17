import pybullet_envs
import gym
import numpy as np
import torch as T

from actor_critic import Agent
from gym import wrappers
import wandb
from tqdm import tqdm

# from tqdm.notebook import tqdm

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
        tags=[config["env_name"]],
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
                    (
                        loss_q,
                        loss_q1,
                        loss_q2,
                        loss_p,
                        log_probs_,
                        action_,
                    ) = agent.learn()
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
                    "score": score,
                    "avg_score": avg_score,
                    "loss_q": loss_q,
                    "loss_q1": loss_q1,
                    "loss_q2": loss_q2,
                    "loss_p": loss_p,
                    "p_log_probs_": log_probs_,
                    "p_action_": action_,
                }
            )

        # WHEN DONE TRIANING TODO: ake it save periodically
        # Save the model in the exchangeable ONNX format
        print("SAVING MODELS AFTER TRAINING..")
        print("OBSERVATION", observation)
        print("\nACTION", action)
        device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        T.onnx.export(
            agent.actor,
            T.tensor(observation, dtype=T.float, device=device),
            "actor.onnx",
        )

        state, action, reward, new_state, done = agent.memory.sample_buffer(
            config.batch_size
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

    # TEST MODE
    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")
