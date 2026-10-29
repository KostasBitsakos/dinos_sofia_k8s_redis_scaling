import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from environment import Environment
from agentLSTM import DDQNAgentLSTM  # Import the LSTM-based agent

# Ensure results directory exists
RESULTS_FOLDER = "final_metrics_lstm"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def evaluate_agent(agent, eval_steps, num_evaluations=5, output_folder=RESULTS_FOLDER):
    """
    Evaluate a pre-trained agent and save metrics.

    Parameters:
    - agent: DDQNAgentLSTM, pre-trained agent to evaluate.
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
            new_state, reward, _, _, _, _, _, done = env.step(action)
            total_reward += reward

            next_state = np.zeros((sequence_length, state_size))
            next_state[:-1] = state[1:]
            next_state[-1] = new_state

            episode_load.append(next_state[-1][0])
            episode_mem_usage.append(next_state[-1][3])
            episode_cpu_usage.append(next_state[-1][2])
            episode_latency.append(next_state[-1][4])
            episode_num_vm.append(next_state[-1][5])

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
    """
    Plots evaluation metrics with a second y-axis showing the number of VMs as dots.
    """
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(12, 10))

    # Load with VMs
    ax1 = plt.subplot(2, 2, 1)
    for episode_load in episode_loads:
        ax1.plot(range(len(episode_load)), episode_load, alpha=0.5, label='Load')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load')
    ax1.set_title('Load Over Time')

    ax1b = ax1.twinx()
    for episode_num_vm in episode_num_vms:
        ax1b.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', alpha=0.3, label='VMs')
    ax1b.set_ylabel('Number of VMs')

    # Memory Usage with VMs
    ax2 = plt.subplot(2, 2, 2)
    for episode_mem in episode_mem_usages:
        ax2.plot(range(len(episode_mem)), episode_mem, alpha=0.5, label='Memory Usage')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Memory Usage')
    ax2.set_title('Memory Usage Over Time')

    ax2b = ax2.twinx()
    for episode_num_vm in episode_num_vms:
        ax2b.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', alpha=0.3, label='VMs')
    ax2b.set_ylabel('Number of VMs')

    # CPU Usage with VMs
    ax3 = plt.subplot(2, 2, 3)
    for episode_cpu in episode_cpu_usages:
        ax3.plot(range(len(episode_cpu)), episode_cpu, alpha=0.5, label='CPU Usage')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('CPU Usage')
    ax3.set_title('CPU Usage Over Time')

    ax3b = ax3.twinx()
    for episode_num_vm in episode_num_vms:
        ax3b.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', alpha=0.3, label='VMs')
    ax3b.set_ylabel('Number of VMs')

    # Latency with VMs
    ax4 = plt.subplot(2, 2, 4)
    for episode_latency in episode_latencies:
        ax4.plot(range(len(episode_latency)), episode_latency, alpha=0.5, label='Latency')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Latency')
    ax4.set_title('Latency Over Time')

    ax4b = ax4.twinx()
    for episode_num_vm in episode_num_vms:
        ax4b.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', alpha=0.3, label='VMs')
    ax4b.set_ylabel('Number of VMs')

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
        model_path = f"modelLSTM_{steps}.pth"

        print(f"🔍 Loading pre-trained model: {model_path}")
        agent = DDQNAgentLSTM(state_size=6, action_size=3, sequence_length=10)
        agent.online_model.load_state_dict(torch.load(model_path))

        print(f"🚀 Evaluating model: {model_path}")
        total_rewards = evaluate_agent(agent, eval_steps, num_evaluations, output_folder=f"{RESULTS_FOLDER}/model_{steps}")

        # Store accumulated reward
        accumulated_rewards.append(sum(total_rewards))

    # Plot accumulated rewards across models
    plot_accumulated_rewards(model_steps, accumulated_rewards)

    print("✅ All evaluations completed! Results are in 'final_metrics_lstm'.")
