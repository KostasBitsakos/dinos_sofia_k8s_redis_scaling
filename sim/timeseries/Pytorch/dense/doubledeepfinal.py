import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from environment import Environment
from ddqnagent import DDQNAgent  # Assuming the Dense model uses the same agent class

# Ensure results directory exists
RESULTS_FOLDER = "final_metrics_dense"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def evaluate_agent(agent, eval_steps, num_evaluations=5, output_folder=RESULTS_FOLDER):
    """
    Evaluate a pre-trained agent and save metrics.

    Parameters:
    - agent: DDQNAgent, pre-trained agent to evaluate.
    - eval_steps: int, maximum number of steps per evaluation.
    - num_evaluations: int, number of evaluation runs.
    - output_folder: str, folder to save plots and metrics.

    Returns:
    - Accumulated rewards and performance metrics.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Initialize environment
    env = Environment()
    state_size = 6
    action_size = 3

    total_rewards = []
    episode_loads = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies = []
    episode_num_vms = []

    for eval_run in range(num_evaluations):
        state = np.zeros(state_size)
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
            new_state, reward, _, _, _, _, _, done = env.step(action)
            total_reward += reward

            # Update state
            next_state = new_state

            # Log metrics
            episode_load.append(next_state[0])   # Load
            episode_mem_usage.append(next_state[2])  # Memory Usage
            episode_cpu_usage.append(next_state[3])  # CPU Usage
            episode_latency.append(next_state[4])  # Latency
            episode_num_vm.append(next_state[5])  # Number of VMs

            state = next_state

            step_count += 1
            if step_count >= eval_steps:
                break

        total_rewards.append(total_reward)
        episode_loads.append(episode_load)
        episode_mem_usages.append(episode_mem_usage)
        episode_cpu_usages.append(episode_cpu_usage)
        episode_latencies.append(episode_latency)
        episode_num_vms.append(episode_num_vm)

    # Save rewards to file
    rewards_file = os.path.join(output_folder, "rewards.txt")
    with open(rewards_file, "w") as f:
        for r in total_rewards:
            f.write(str(r) + "\n")

    # Generate and save plots
    plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_folder)

    avg_reward = np.mean(total_rewards)
    print(f"✅ Evaluation complete. Average reward: {avg_reward}")
    return total_rewards

def plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(12, 10))

    # Consistent Colors Across All Models
    load_color = 'blue'
    memory_color = 'green'
    cpu_color = 'orange'
    latency_color = 'purple'
    vm_color = 'red'  # VMs represented by red dots

    def plot_subplot(ax, data, title, ylabel, color, episode_num_vms):
        for episode in data:
            ax.plot(range(len(episode)), episode, color=color, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel, color=color)

        ax2 = ax.twinx()
        for vms in episode_num_vms:
            ax2.scatter(range(len(vms)), vms, color=vm_color, marker='o', alpha=0.3)
        ax2.set_ylabel("Number of VMs", color=vm_color)

    # Load Plot
    plot_subplot(plt.subplot(2, 2, 1), episode_loads, "Load Over Time", "Load", load_color, episode_num_vms)

    # Memory Usage Plot
    plot_subplot(plt.subplot(2, 2, 2), episode_mem_usages, "Memory Usage Over Time", "Memory Usage", memory_color, episode_num_vms)

    # CPU Usage Plot
    plot_subplot(plt.subplot(2, 2, 3), episode_cpu_usages, "CPU Usage Over Time", "CPU Usage", cpu_color, episode_num_vms)

    # Latency Plot
    plot_subplot(plt.subplot(2, 2, 4), episode_latencies, "Latency Over Time", "Latency", latency_color, episode_num_vms)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/evaluation_metrics.jpg")
    plt.close()


def plot_accumulated_rewards(model_steps, accumulated_rewards, output_folder=RESULTS_FOLDER):
    """
    Plots the accumulated rewards for different models.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(model_steps, accumulated_rewards, color='blue', alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Total Accumulated Reward")
    plt.title("Accumulated Rewards for Models Trained with Different Steps")
    plt.xticks(model_steps, [f"Model {s}" for s in model_steps])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{output_folder}/accumulated_rewards.jpg")
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    eval_steps = 5000
    num_evaluations = 5

    # Define models to evaluate
    model_steps = [1, 2, 5, 10, 20]  
    accumulated_rewards = []

    for steps in model_steps:
        model_path = f"modelCPU_{steps}.pth"  # ✅ Ensuring correct model names

        print(f"🔍 Loading pre-trained model: {model_path}")
        agent = DDQNAgent(state_size=6, action_size=3)
        agent.online_model.load_state_dict(torch.load(model_path))

        print(f"🚀 Evaluating model: {model_path}")
        total_rewards = evaluate_agent(agent, eval_steps, num_evaluations, output_folder=f"{RESULTS_FOLDER}/model_{steps}")

        # Store accumulated reward
        accumulated_rewards.append(sum(total_rewards))

    # Plot accumulated rewards across models
    plot_accumulated_rewards(model_steps, accumulated_rewards)

    print("✅ All evaluations completed! Results are in 'final_metrics_dense'.")
