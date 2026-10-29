import numpy as np
import os
import matplotlib.pyplot as plt
from environment import Environment
from simpleagent import SimpleAgent

# Ensure results directory exists
RESULTS_FOLDER = "final_metrics_hpa"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def evaluate_agent(agent, eval_steps, num_evaluations=5, output_folder=RESULTS_FOLDER):
    """
    Evaluate the HPA agent and save metrics.

    Parameters:
    - agent: SimpleAgent, rule-based autoscaler.
    - eval_steps: int, number of evaluation steps.
    - num_evaluations: int, number of runs.
    - output_folder: str, folder to save plots and metrics.

    Returns:
    - Accumulated rewards and performance metrics.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Initialize environment
    env = Environment()
    total_rewards = []
    episode_loads = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies = []
    episode_num_vms = []

    for eval_run in range(num_evaluations):
        state = env.reset()
        total_reward = 0

        episode_load = []
        episode_mem_usage = []
        episode_cpu_usage = []
        episode_latency = []
        episode_num_vm = []

        done = False
        step_count = 0
        while not done:
            action = agent.act(state)
            new_state, reward, _, _, _, _, _, done = env.step(action)
            total_reward += reward

            # Log metrics
            episode_load.append(new_state[0])   # Load
            episode_mem_usage.append(new_state[2])  # Memory Usage
            episode_cpu_usage.append(new_state[3])  # CPU Usage
            episode_latency.append(new_state[4])  # Latency
            episode_num_vm.append(new_state[5])  # Number of VMs

            state = new_state

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
    print(f"✅ HPA Evaluation complete for {output_folder}. Average reward: {avg_reward}")
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
    Plots the accumulated rewards for different runs of the HPA agent.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(model_steps, accumulated_rewards, color='blue', alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Total Accumulated Reward")
    plt.title("Accumulated Rewards for HPA Across Different Steps")
    plt.xticks(model_steps, [f"HPA {s} steps" for s in model_steps])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{output_folder}/accumulated_rewards.jpg")
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    eval_steps = 5000
    num_evaluations = 5
    model_steps = [1, 2, 5, 10, 20]  
    accumulated_rewards = []

    agent = SimpleAgent()

    for steps in model_steps:
        model_folder = os.path.join(RESULTS_FOLDER, f"model_{steps}")
        print(f"🚀 Evaluating HPA for {steps} training steps...")

        total_rewards = evaluate_agent(agent, eval_steps, num_evaluations, output_folder=model_folder)

        # Store accumulated reward
        accumulated_rewards.append(sum(total_rewards))

    # Plot accumulated rewards across different steps
    plot_accumulated_rewards(model_steps, accumulated_rewards)

    print("✅ HPA evaluation for all steps completed! Results are in 'final_metrics_hpa'.")
