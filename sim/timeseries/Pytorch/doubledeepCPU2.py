import numpy as np
import os
import torch
from environmentCPU2 import Environment
from agentHuber import DDQNAgent
import matplotlib.pyplot as plt


def fine_tune_agent(pretrained_model_path, fine_tune_steps, batch_size=16, output_folder="new_metrics"):
    """
    Fine-tune a pre-trained agent on the updated environment.

    Parameters:
    - pretrained_model_path: str, path to the pre-trained model file.
    - fine_tune_steps: int, number of steps for fine-tuning.
    - batch_size: int, size of the replay batch.
    - output_folder: str, folder to save metrics and losses.
    """
    env = Environment()
    state_size = 6
    action_size = 3
    sequence_length = 10

    # Load pre-trained model
    agent = DDQNAgent(state_size, action_size, sequence_length)
    agent.online_model.load_state_dict(torch.load(pretrained_model_path))

    # Fine-tuning loop
    state = np.zeros((sequence_length, state_size))
    os.makedirs(output_folder, exist_ok=True)
    loss_file = os.path.join(output_folder, f"losses_{pretrained_model_path}.txt")

    for step in range(fine_tune_steps):
        action = agent.act(state)
        next_state, reward, _, _, _, _, _, done = env.step(action)

        next_state_sequence = np.zeros((sequence_length, state_size))
        next_state_sequence[:-1] = state[1:]
        next_state_sequence[-1] = next_state

        agent.remember(state, action, reward, next_state_sequence, done)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # Save loss to file
        with open(loss_file, "a") as f:
            if len(agent.loss_history) > 0:
                f.write(f"{agent.loss_history[-1]}\n")

        state = next_state_sequence
        if done:
            break

    # Save fine-tuned model
    fine_tuned_model_path = pretrained_model_path.replace(".pth", "_fine_tuned.pth")
    agent.save_model(fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")


def evaluate_agent(agent, eval_steps, num_evaluations=5, output_folder="new_metrics"):
    """
    Evaluate a pre-trained agent on the updated environment and save metrics.

    Parameters:
    - agent: DDQNAgent, pre-trained agent to evaluate.
    - eval_steps: int, maximum number of steps per evaluation.
    - num_evaluations: int, number of evaluation runs.
    - output_folder: str, folder to save plots and metrics.

    Returns:
    - Metrics for evaluations.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Initialize environment and metrics
    env = Environment()
    state_size = 6
    action_size = 3
    sequence_length = 10

    total_rewards = []
    episode_loads = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies = []
    episode_num_vms = []

    for eval_run in range(num_evaluations):
        state = np.zeros((sequence_length, state_size))
        total_reward = 0

        episode_load = []
        episode_mem_usage = []
        episode_cpu_usage = []
        episode_latency = []
        episode_num_vm = []

        done = False
        step_count = 0
        while not done:
            action = agent.act(state, verbose=0)

            # Step in the updated environment
            new_state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, done = env.step(action)
            total_reward += reward

            # Update state sequence
            next_state = np.zeros((sequence_length, state_size))
            next_state[:-1] = state[1:]
            next_state[-1] = new_state

            # Log metrics
            episode_load.append(next_state[-1][0])
            episode_mem_usage.append(next_state[-1][3])
            episode_cpu_usage.append(next_state[-1][2])
            episode_latency.append(next_state[-1][4])
            episode_num_vm.append(next_state[-1][5])

            state = next_state

            step_count += 1
            if step_count >= eval_steps:
                break

        # Save metrics for this run
        total_rewards.append(total_reward)
        episode_loads.append(episode_load)
        episode_mem_usages.append(episode_mem_usage)
        episode_cpu_usages.append(episode_cpu_usage)
        episode_latencies.append(episode_latency)
        episode_num_vms.append(episode_num_vm)

    # Save rewards to file
    rewards_file = os.path.join(output_folder, "rewards.txt")
    with open(rewards_file, "a") as f:
        f.write(f"{total_rewards}\n")

    # Plot results
    plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix=output_folder)

    # Print summary
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_evaluations} evaluations: {avg_reward}")

    return total_rewards


def plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix):
    """
    Plots the metrics over time, overlaying the normalized number of VMs.

    Parameters:
    - output_filename_prefix: str, folder path where the plot will be saved.
    """
    import matplotlib.pyplot as plt
    os.makedirs(output_filename_prefix, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Normalize VM counts
    max_vms = 20
    normalized_episode_vms = [[vm / max_vms for vm in episode] for episode in episode_num_vms]

    # Plot Load and Normalized VMs
    ax1 = plt.subplot(2, 2, 1)
    for episode_load, episode_vm in zip(episode_loads, normalized_episode_vms):
        ax1.plot(range(len(episode_load)), episode_load, alpha=0.5, label='Load')
        ax1.plot(range(len(episode_vm)), episode_vm, linestyle='--', alpha=0.7, label='Normalized Num VMs')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Load / Normalized Num VMs')
    ax1.set_title('Load vs Normalized Num VMs Over Time')
    ax1.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_filename_prefix}/metrics_with_vms.jpg")
    plt.close()


if __name__ == "__main__":
    fine_tune_steps = 1000  
    eval_steps = 5000  
    num_evaluations = 5  
    batch_size = 16  
    metrics_folder = "new_metrics"  

    model_steps = [1, 2, 5, 10, 20]  
    final_rewards = {}

    for steps in model_steps:
        pretrained_model_path = f"modelCPU_{steps}.pth"

        fine_tune = True  
        if fine_tune:
            print(f"Fine-tuning the model: {pretrained_model_path}")
            fine_tune_agent(pretrained_model_path, fine_tune_steps, batch_size, output_folder=f"{metrics_folder}/model_{steps}")
            pretrained_model_path = pretrained_model_path.replace(".pth", "_fine_tuned.pth")

        print(f"Evaluating the model: {pretrained_model_path}")
        agent = DDQNAgent(state_size=6, action_size=3, sequence_length=10)
        agent.online_model.load_state_dict(torch.load(pretrained_model_path))

        rewards = evaluate_agent(agent, eval_steps, num_evaluations, output_folder=f"{metrics_folder}/model_{steps}")
        final_rewards[steps] = np.mean(rewards)

    os.makedirs(metrics_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))

    plt.bar(final_rewards.keys(), final_rewards.values(), tick_label=[f"Model {s}" for s in final_rewards.keys()], color='blue')
    plt.xlabel("Training Steps")
    plt.ylabel("Average Reward")
    plt.title("Reward Comparison for Different Training Steps")

    final_plot_path = os.path.join(metrics_folder, "final_rewards_comparison.jpg")
    plt.savefig(final_plot_path)
    plt.close()

    print(f"Final reward comparison plot saved in {final_plot_path}.")
