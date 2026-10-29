import numpy as np
import os
import sys
import torch
import time

import matplotlib.pyplot as plt
from environment import Environment
from agentHuber import DDQNAgentLSTM
import runctl

# Define results folder (default; see _out())
RESULTS_FOLDER = "final_quantum_compare_metrics_lstm"

# Fine-tuning with time hooks clearly
def _out():
    return runctl.results_folder(RESULTS_FOLDER)


def fine_tune_agent(agent, env, fine_tune_steps=1000, batch_size=16):
    print(f"\n🔄 Fine-tuning agent for {fine_tune_steps} steps...")
    start_time = time.time()
    agent.fine_tune(env, fine_tune_steps, batch_size)
    elapsed = time.time() - start_time

    fine_tuned_model_path = f"{_out()}/{agent.pretrained_model.replace('.pth', '_fine_tuned.pth')}"
    agent.save_model(fine_tuned_model_path)
    print(f"✅ Fine-tuned model saved to {fine_tuned_model_path}")
    print(f"⏱️ Fine-tuning completed in {elapsed:.2f}s")

    with open(f"{_out()}/timing_results.txt", "a") as f:
        f.write(f"Fine-tuning time ({agent.pretrained_model}): {elapsed:.2f} seconds\n")


def evaluate_agent(agent, env, eval_steps=5000, num_evaluations=5, output_folder=None, model=None, episode_offset=0, total_episodes=None):
    output_folder = output_folder or _out()
    os.makedirs(output_folder, exist_ok=True)

    total_rewards = []
    episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms = [], [], [], [], []

    start_time = time.time()
    for eval_run in range(num_evaluations):
        print(f"\n▶️ Starting evaluation run {eval_run + 1}/{num_evaluations}")
        state = env.reset()
        state = np.zeros((agent.sequence_length, agent.state_size))
        total_reward = 0

        episode_load, episode_mem_usage, episode_cpu_usage, episode_latency, episode_num_vm = [], [], [], [], []

        for step in range(eval_steps):
            action = agent.act(state)
            new_state, reward, _, _, _, _, _, done = env.step(action)
            total_reward += reward

            next_state = np.zeros((agent.sequence_length, agent.state_size))
            next_state[:-1] = state[1:]
            next_state[-1] = new_state

            episode_load.append(new_state[0])
            episode_mem_usage.append(new_state[3])
            episode_cpu_usage.append(new_state[2])
            episode_latency.append(new_state[4])
            episode_num_vm.append(new_state[5])

            state = next_state

        total_rewards.append(total_reward)
        runctl.log_metrics(step=episode_offset + eval_run, model=model, reward=float(total_reward),
                           mean_vms=float(np.mean(episode_num_vm)))
        runctl.write_progress(episode_offset + eval_run + 1, total_episodes or num_evaluations)
        episode_loads.append(episode_load)
        episode_mem_usages.append(episode_mem_usage)
        episode_cpu_usages.append(episode_cpu_usage)
        episode_latencies.append(episode_latency)
        episode_num_vms.append(episode_num_vm)

    elapsed = time.time() - start_time

    rewards_file = os.path.join(output_folder, "rewards.txt")
    with open(rewards_file, "w") as f:
        for r in total_rewards:
            f.write(str(r) + "\n")

    plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_folder)

    avg_reward = np.mean(total_rewards)
    print(f"✅ Evaluation complete. Average reward: {avg_reward}")
    print(f"⏱️ Evaluation completed in {elapsed:.2f}s")

    with open(f"{_out()}/timing_results.txt", "a") as f:
        f.write(f"Evaluation time ({agent.pretrained_model}): {elapsed:.2f} seconds\n")

    return total_rewards


def plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(12, 10))

    # Define consistent colors across all plots
    load_color = 'blue'
    memory_color = 'green'
    cpu_color = 'orange'
    latency_color = 'purple'
    vm_color = 'red'  # VMs as dots

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


def plot_accumulated_rewards(model_steps, accumulated_rewards, output_folder=None):
    output_folder = output_folder or _out()
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.bar(model_steps, accumulated_rewards, color='blue', alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Total Accumulated Reward")
    plt.title("Accumulated Rewards for Models Trained with Different Steps")
    plt.xticks(model_steps, [f"Model {s}" for s in model_steps])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{output_folder}/accumulated_rewards.jpg")
    plt.close()

def legacy_from_env(argv=None):
    """QRL_LEGACY=1 or --legacy: reproduce the published (buggy) behaviour."""
    argv = sys.argv[1:] if argv is None else argv
    return os.environ.get("QRL_LEGACY", "0") == "1" or "--legacy" in argv


# --- MAIN EXECUTION ---
def main(legacy=None, model_steps=(1, 2, 5, 10, 20), eval_steps=5000, num_evaluations=5, fine_tune_steps=1000, seed=None):
    if legacy is None:
        legacy = legacy_from_env()
    os.makedirs(_out(), exist_ok=True)
    runctl.seed_everything(seed)
    model_steps = tuple(model_steps)
    accumulated_rewards = []

    env = Environment()

    total_exp_start = time.time()
    for idx, steps in enumerate(model_steps):
        model_path = f"modelCPU_{steps}.pth"

        print(f"\n🔍 Loading pre-trained model: {model_path}")
        agent = DDQNAgentLSTM(state_size=6, action_size=3, sequence_length=10, pretrained_model=model_path, legacy=legacy)

        fine_tune_agent(agent, env, fine_tune_steps=fine_tune_steps, batch_size=16)

        total_rewards = evaluate_agent(agent, env, eval_steps, num_evaluations, output_folder=f"{_out()}/model_{steps}",
                                     model=steps, episode_offset=idx * num_evaluations,
                                     total_episodes=len(model_steps) * num_evaluations)

        accumulated_rewards.append(sum(total_rewards))

    total_exp_elapsed = time.time() - total_exp_start
    plot_accumulated_rewards(list(model_steps), accumulated_rewards, _out())
    print(f"\n✅ All evaluations completed in {total_exp_elapsed:.2f}s! Results are in '{_out()}'.")

    with open(f"{_out()}/timing_results.txt", "a") as f:
        f.write(f"Total experiment time: {total_exp_elapsed:.2f} seconds\n")


if __name__ == "__main__":
    a = runctl.parse_args()
    main(legacy=legacy_from_env(), model_steps=a.model_steps, eval_steps=a.eval_steps,
         num_evaluations=a.num_evaluations, fine_tune_steps=a.fine_tune_steps, seed=a.seed)
