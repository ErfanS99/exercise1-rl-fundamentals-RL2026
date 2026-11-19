"""
    Robot Learning
    Exercise 1

    Reinforcement Learning 

    Polito A-Y 2026-2027
"""
import torch
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim

import sys
from pathlib import Path
import json
import shutil


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--central_point", type=float, default=0.0,
                        help="Point x0 to fluctuate around")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                    help="Learning rate for the Adam optimizer")

    parser.add_argument("--max_episode_steps", type=int, default=200,
                        help="Maximum number of timesteps per episode")

    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        help="Name of the reward-function experiment"
    )

    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,
          silent=False, train_run_id=0, x0=0, random_policy=False):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation, _ = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                # Task 1.1
                """
                Sample a random action from the action space
                """
                action = env.action_space.sample()

            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            reward = new_reward(observation, x0)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False, x0=0):
    test_reward, test_len = 0, 0

    # episodes = 100
    print('Num testing episodes:', episodes)

    for ep in range(episodes):
        done = False
        observation, _ = env.reset()
        while not done:
        # Task 1.2
            """
            Test on 500 timesteps
            """
            # TODO

            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            # reward = new_reward(observation, x0)

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)






def new_reward(state, x0):
    x = state[0]
    x_dot = state[1]
    theta = state[2]

    # Reward Function 1, Version 6:
    # Balance the pole and return faster to x = 0

    # Desired velocity toward the center
    desired_velocity = np.clip(-1.0 * x, -1.0, 1.0)

    # Position regulation has higher priority
    position_penalty = 3.0 * (x / 2.4)**2

    # Allow some pole inclination for cart acceleration
    angle_penalty = 0.2 * (theta / 0.2095)**2

    # Encourage motion toward the center, then slowing down
    velocity_penalty = 0.2 * ((x_dot - desired_velocity) / 2.0)**2

    reward = 1.0 - position_penalty - angle_penalty - velocity_penalty

    return reward





# The main function
def main(args):
    # Create a Gymnasium environment (CartPole-v1)
    render_mode = "rgb_array" if args.render_training or args.render_test else None

    # Create the environment
    render_mode = "human" if (args.render_training or args.render_test) else None
    env = gym.make(args.env, render_mode=render_mode)


    # Task 1.2
    """
    # For CartPole-v1 - change the maximum episode length to 1000
    """
    # TODO
    env._max_episode_steps = args.max_episode_steps

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, learning_rate=args.learning_rate)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        # Train
        # Create a unique folder for this experiment
        experiment_dir = Path("experiments") / args.experiment

        if experiment_dir.exists():
            raise FileExistsError(
                f"Experiment '{args.experiment}' already exists. "
                "Choose a new experiment name."
            )

        experiment_dir.mkdir(parents=True)
        training_history = train(agent, env, args.train_episodes, False, args.render_training, x0=args.central_point, random_policy=args.random_policy)

        # Save the model
        model_file = experiment_dir / "model.ai"
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
        sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
        plt.legend()

        plt.title("Reward history (%s)" % args.env)

        # Save the training plot before displaying it
        plt.savefig(
            experiment_dir / "training.png",
            dpi=150,
            bbox_inches="tight"
        )

        # Save experiment configuration
        with open(experiment_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # Preserve the exact code used for this experiment
        shutil.copy2(
            Path(__file__),
            experiment_dir / "cartpole_rl.py"
        )

        plt.show()

        print("Training finished.")
    else:
        # Test
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args.render_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

