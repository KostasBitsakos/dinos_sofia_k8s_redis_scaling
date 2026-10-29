import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import time  # Added for timing
from environment import Environment
from agentHuber import DDQNAgentLSTM
import pennylane as qml  

# Quantum setup
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def quantum_sampler(state_embedding):
    qml.AngleEmbedding(state_embedding, wires=[0,1])
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0,1])

# Define results folder
RESULTS_FOLDER = "final_metrics_lstm_quantum"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def fine_tune_agent(agent, env, fine_tune_steps=1000, batch_size=16):
    print(f"\n🔄 Fine-tuning agent for {fine_tune_steps} steps...")
    
    start_time = time.time()
    agent.fine_tune(env, fine_tune_steps, batch_size)
    elapsed = time.time() - start_time
    
    fine_tuned_model_path = f"{RESULTS_FOLDER}/{agent.pretrained_model.replace('.pth', '_fine_tuned.pth')}"
    agent.save_model(fine_tuned_model_path)
    
    print(f"✅ Fine-tuned model saved to {fine_tuned_model_path}")
    print(f"⏱️ Fine-tuning completed in {elapsed:.2f}s")
    
    # Save fine-tuning time clearly
    with open(f"{RESULTS_FOLDER}/timing_results.txt", "a") as f:
        f.write(f"Fine-tuning time ({agent.pretrained_model}): {elapsed:.2f} seconds\n")


def quantum_act(agent, state):
    if np.random.rand() <= agent.epsilon:
        state_embedding = state[-1][:2]
        probs = quantum_sampler(state_embedding)
        probs = probs[:agent.action_size]
        probs /= probs.sum()  # clearly normalize probabilities
        action = np.random.choice(agent.action_size, p=probs)
        return action
    else:
        return agent.act(state)


def evaluate_agent(agent, env, eval_steps=5000, num_evaluations=5, output_folder=RESULTS_FOLDER):
    os.makedirs(output_folder, exist_ok=True)

    total_rewards = []
    episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms = [], [], [], [], []

    start_time = time.time()
    for eval_run in range(num_evaluations):
        print(f"\n▶️ Starting evaluation run {eval_run + 1}/{num_evaluations}")
        env.reset()
        state = np.zeros((agent.sequence_length, agent.state_size))
        total_reward = 0

        episode_load, episode_mem_usage, episode_cpu_usage, episode_latency, episode_num_vm = [], [], [], [], []

        for step in range(eval_steps):
            action = quantum_act(agent, state)
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

    # Save evaluation time clearly
    with open(f"{RESULTS_FOLDER}/timing_results.txt", "a") as f:
        f.write(f"Evaluation time ({agent.pretrained_model}): {elapsed:.2f} seconds\n")

    return total_rewards


def plot_metrics(episode_loads, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(12, 10))

    load_color = 'blue'
    memory_color = 'green'
    cpu_color = 'orange'
    latency_color = 'purple'
    vm_color = 'red'

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

    plot_subplot(plt.subplot(2, 2, 1), episode_loads, "Load Over Time", "Load", load_color, episode_num_vms)
    plot_subplot(plt.subplot(2, 2, 2), episode_mem_usages, "Memory Usage Over Time", "Memory Usage", memory_color, episode_num_vms)
    plot_subplot(plt.subplot(2, 2, 3), episode_cpu_usages, "CPU Usage Over Time", "CPU Usage", cpu_color, episode_num_vms)
    plot_subplot(plt.subplot(2, 2, 4), episode_latencies, "Latency Over Time", "Latency", latency_color, episode_num_vms)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/evaluation_metrics.jpg")
    plt.close()

def plot_accumulated_rewards(model_steps, accumulated_rewards, output_folder=RESULTS_FOLDER):
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
    model_steps = [1, 2, 5, 10, 20]
    accumulated_rewards = []

    env = Environment()

    total_exp_start = time.time()
    for steps in model_steps:
        model_path = f"modelCPU_{steps}.pth"

        print(f"\n🔍 Loading pre-trained model: {model_path}")
        agent = DDQNAgentLSTM(state_size=6, action_size=3, sequence_length=10, pretrained_model=model_path)

        fine_tune_agent(agent, env, fine_tune_steps=1000, batch_size=16)

        total_rewards = evaluate_agent(agent, env, eval_steps, num_evaluations, output_folder=f"{RESULTS_FOLDER}/model_{steps}")

        accumulated_rewards.append(sum(total_rewards))

    total_exp_elapsed = time.time() - total_exp_start
    plot_accumulated_rewards(model_steps, accumulated_rewards, RESULTS_FOLDER)
    print(f"\n✅ All quantum-enhanced evaluations completed in {total_exp_elapsed:.2f}s! Results are in '{RESULTS_FOLDER}'.")

    # Save total experiment time clearly
    with open(f"{RESULTS_FOLDER}/timing_results.txt", "a") as f:
        f.write(f"Total experiment time: {total_exp_elapsed:.2f} seconds\n")

