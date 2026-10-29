import numpy as np
import os
import torch
from environment2 import Environment
from ddqnagent2 import DDQNAgent
import matplotlib.pyplot as plt

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

    env = Environment()
    total_rewards = []
    episode_loads = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies = []
    episode_num_vms = []

    for eval_run in range(num_evaluations):
        state = np.zeros((6,))
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

            new_state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, done = env.step(action)
            total_reward += reward

            next_state = new_state

            episode_load.append(next_state[0])
            episode_mem_usage.append(next_state[2])  # Memory
            episode_cpu_usage.append(next_state[1])  # CPU
            episode_latency.append(next_state[3])    # Latency
            episode_num_vm.append(next_state[5])     # Number of VMs

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

    # Save rewards
    rewards_file = os.path.join(output_folder, "rewards.txt")
    with open(rewards_file, "a") as f:
        f.write(f"{total_rewards}\n")

    # Plot metrics
    plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix=output_folder)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_evaluations} evaluations: {avg_reward}")
    return episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, total_rewards

def plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix):
    os.makedirs(output_filename_prefix, exist_ok=True)
    plt.figure(figsize=(12, 8))

    # Plot Load
    plt.subplot(2, 2, 1)
    for episode_load in episode_loads:
        plt.plot(range(len(episode_load)), episode_load, alpha=0.5, label='Load')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.title('Load Over Time')
    plt.legend()

    # Plot Memory Usage
    plt.subplot(2, 2, 2)
    for episode_mem_usage in episode_mem_usages:
        plt.plot(range(len(episode_mem_usage)), episode_mem_usage, alpha=0.5, label='Memory Usage')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage')
    plt.title('Memory Usage Over Time')
    plt.legend()

    # Plot CPU Usage
    plt.subplot(2, 2, 3)
    for episode_cpu_usage in episode_cpu_usages:
        plt.plot(range(len(episode_cpu_usage)), episode_cpu_usage, alpha=0.5, label='CPU Usage')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage')
    plt.title('CPU Usage Over Time')
    plt.legend()

    # Plot Latency
    plt.subplot(2, 2, 4)
    for episode_latency in episode_latencies:
        plt.plot(range(len(episode_latency)), episode_latency, alpha=0.5, label='Latency')
    plt.xlabel('Time')
    plt.ylabel('Latency')
    plt.title('Latency Over Time')
    plt.legend()

    # Save plots
    plt.tight_layout()
    plt.savefig(f"{output_filename_prefix}/metrics.jpg")
    plt.close()

if __name__ == "__main__":
    fine_tune_steps = 1000  
    eval_steps = 5000  
    model_steps = [1, 2, 5, 10, 20]  
    output_folder = "new_metrics"  

    for steps in model_steps:
        model_path = f"modelCPU_{steps}.pth"  
        agent = DDQNAgent(state_size=6, action_size=3)

        print(f"Fine-tuning model: {model_path}")
        agent.fine_tune(model_path, fine_tune_steps, output_folder=f"{output_folder}/model_{steps}")

        # **Evaluate the fine-tuned model**
        fine_tuned_model_path = model_path.replace(".pth", "_fine_tuned.pth")
        print(f"Evaluating model: {fine_tuned_model_path}")

        agent.online_model.load_state_dict(torch.load(fine_tuned_model_path))
        evaluate_agent(agent, eval_steps, output_folder=f"{output_folder}/model_{steps}")

    print("All evaluations completed. Check the 'new_metrics' folder for results.")
