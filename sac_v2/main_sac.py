import pybullet_envs
import gym
import numpy as np
import torch as T
import time

from actor_critic import Agent
from gym import wrappers
import wandb
from tqdm import tqdm


def save_model(agent, observation):
    # Save the model in the exchangeable ONNX format
    print("SAVING MODELS AFTER TRAINING..")
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    T.onnx.export(
        agent.actor, T.tensor(observation, dtype=T.float, device=device), "actor.onnx",
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


def main(config):
    env = gym.make(config["env_name"])
    # record videos of the agent playing the game
    env = gym.wrappers.Monitor(
        env,
        "tmp/video",
        video_callable=(lambda episode_id: episode_id % 10 == 0),
        force=True,
    )

    epochs = config["epochs"]
    update_every = config["update_every"]
    max_ep_len = config["max_ep_len"]
    start_steps = config["start_steps"]
    steps_per_epoch = config["steps_per_epoch"]
    update_after = config["update_after"]

    with wandb.init(
        project="trashbot-sac",
        tags=[config["env_name"], "v2"],
        config=config,
        monitor_gym=True,
    ):
        config = wandb.config

        agent = Agent(
            env=env,
            input_dims=env.observation_space.shape,
            act_dims=env.action_space.shape[0],
            act_limit=env.action_space.high[0],
            alpha=config["alpha"],
            gamma=config["gamma"],
            max_size=config["max_size"],
            tau=config["tau"],
            lr=config["lr"],
            layer1_size=config["layer1_size"],
            layer2_size=config["layer2_size"],
            batch_size=config["batch_size"],
        )

        wandb.watch(
            [agent.actor, agent.critic_1, agent.critic_2,], log="all", log_freq=10,
        )

        # load_checkpoint = False
        total_steps = steps_per_epoch * epochs
        observation, ep_ret, ep_len = env.reset(), 0, 0
        start_time = time.time()

        # Main loop: collect experience in env and update/log each epoch
        for i in tqdm(range(total_steps)):

            # Epsisode  # while not done:
            if i > start_steps:
                action = agent.choose_action(observation)
            else:
                action = env.action_space.sample()

            observation_, reward, done, info = env.step(action)
            ep_len += 1
            ep_ret += reward

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if ep_len == max_ep_len else done

            agent.remember(observation, action, reward, observation_, done)

            observation = observation_

            # end of episode reset
            if done or (ep_len == max_ep_len):
                logs = {
                    "ep_len": ep_len,
                    "ep_ret": ep_ret,
                }
                print(f"\nLogs: {logs}")
                wandb.log(logs)
                observation, ep_ret, ep_len = env.reset(), 0, 0

            # update agent
            if i >= update_after and i % update_every == 0:
                for j in range(update_every):
                    (q_info, pi_info, loss_q, loss_q1, loss_q2, loss_p,) = agent.learn()

            # end of epoch handling
            # TODO: save_model(agent, observation)
            if (i + 1) % steps_per_epoch == 0:

                epoch = (i + 1) // steps_per_epoch

                ep_logs = {
                    "TotalEnvInteracts": i,
                    "epoch": epoch,
                    "score": ep_ret,
                    "loss_q": loss_q,
                    "loss_q1": loss_q1,
                    "loss_q2": loss_q2,
                    "loss_p": loss_p,
                    "q_info": q_info,
                    "pi_info": pi_info,
                    "env_info": info,
                    "ep_time": time.time() - start_time,
                }
                print(f"\n{ep_logs}\n")
                wandb.log(ep_logs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetahBulletEnv-v0")
    # parser.add_argument("--n_games", type=int, default=1000)
    args = parser.parse_args()

    config = dict(
        # n_games=args.n_games,
        # env_name="InvertedPendulumBulletEnv-v0",
        # env_name="AntBulletEnv-v0",
        # env_name="LunarLanderContinuous-v2",
        env_name=args.env,
        alpha=0.4,  # 1/reward_scale
        gamma=0.99,
        max_size=1_000_000,
        tau=0.005,
        lr=1e-3,  # instead of 3e-4
        layer1_size=256,
        layer2_size=256,
        batch_size=100,
        start_steps=10000,
        steps_per_epoch=4000,
        epochs=200,
        update_every=50,
        max_ep_len=1000,  # dont' change usually depends on env time limit
        update_after=1000,
    )
    main(config)
