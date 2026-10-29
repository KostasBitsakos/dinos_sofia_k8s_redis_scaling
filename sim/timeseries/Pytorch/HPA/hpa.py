import numpy as np
import pandas as pd
from environment import Environment
from simpleagent import SimpleAgent
import matplotlib.pyplot as plt

def train_agent(train_steps, max_episodes_per_step=1000):
    env = Environment()
    agent = SimpleAgent()

    rewards = []
    episode_loads = []
    episode_read_percentages = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies = []
    episode_num_vms = []

    for _ in range(train_steps):
        state = env.reset()
        total_reward = 0

        episode_load = []
        episode_read_percentage = []
        episode_mem_usage = []
        episode_cpu_usage = []
        episode_latency = []
        episode_num_vm = []
        
        done = False
        step_count = 0
        from tqdm import tqdm
        for episode_idx in tqdm(range(max_episodes_per_step)):
            action = agent.act(state)
            new_state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, done = env.step(action)
            total_reward += reward
                
            episode_load.append(new_state[0])
            episode_read_percentage.append(new_state[1])
            episode_mem_usage.append(new_state[2])
            episode_cpu_usage.append(new_state[3])
            episode_latency.append(new_state[4])
            episode_num_vm.append(new_state[5])

            state = new_state

            if done:
                break
            step_count += 1

        rewards.append(total_reward)
        episode_loads.append(episode_load)
        episode_read_percentages.append(episode_read_percentage)
        episode_mem_usages.append(episode_mem_usage)
        episode_cpu_usages.append(episode_cpu_usage)
        episode_latencies.append(episode_latency)
        episode_num_vms.append(episode_num_vm)
        if done:
            break

    # Plot training rewards
    plt.figure()
    plt.plot(range(len(rewards)), rewards, label='Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Training Progress for {train_steps} Steps')
    plt.legend()
    plt.savefig(f'training_rewards_{train_steps}.jpg')
    plt.close()
    with open(f"rewards_{train_steps}.txt", "w") as f:
        for reward in rewards:
            f.write(str(reward) + "\n")

    return episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, rewards


def evaluate_agent(agent, eval_steps, num_evaluations=5):
    env = Environment()
    total_rewards = []
    episode_loads = []
    episode_read_percentages = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies = []
    episode_num_vms = []

    for _ in range(num_evaluations):  # Run multiple evaluations
        state = env.reset()
        total_reward = 0

        episode_load = []
        episode_read_percentage = []
        episode_mem_usage = []
        episode_cpu_usage = []
        episode_latency = []
        episode_num_vm = []

        done = False
        idx = 0
        while not done:
            idx += 1
            action = agent.act(state)
            new_state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, done = env.step(action)
            total_reward += reward

            episode_load.append(new_state[0])
            episode_read_percentage.append(new_state[1])
            episode_mem_usage.append(new_state[2])
            episode_cpu_usage.append(new_state[3])
            episode_latency.append(new_state[4])
            episode_num_vm.append(new_state[5])

            state = new_state

            if idx > eval_steps:
                break

        total_rewards.append(total_reward)
        episode_loads.append(episode_load)
        episode_read_percentages.append(episode_read_percentage)
        episode_mem_usages.append(episode_mem_usage)
        episode_cpu_usages.append(episode_cpu_usage)
        episode_latencies.append(episode_latency)
        episode_num_vms.append(episode_num_vm)

    avg_reward = np.mean(total_rewards)
    print(f"Avg. reward for evaluation: {avg_reward}")

    return episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, total_rewards

def plot_metrics(episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix):
    plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(3, 2, 1)
    for episode_load in episode_loads:
        ax1.plot(range(len(episode_load)), episode_load, alpha=0.5, label='Load')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load')
    ax1.set_title('Load Over Time')

    ax2 = ax1.twinx()
    for episode_num_vm in episode_num_vms:
        ax2.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', label='Num VMs')
    ax2.set_ylabel('Number of VMs')

    ax3 = plt.subplot(3, 2, 2)
    for episode_read_percentage in episode_read_percentages:
        ax3.plot(range(len(episode_read_percentage)), episode_read_percentage, alpha=0.5, label='Read Percentage')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Read Percentage')
    ax3.set_title('Read Percentage Over Time')

    ax4 = ax3.twinx()
    for episode_num_vm in episode_num_vms:
        ax4.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', label='Num VMs')
    ax4.set_ylabel('Number of VMs')

    ax5 = plt.subplot(3, 2, 3)
    for episode_mem_usage in episode_mem_usages:
        ax5.plot(range(len(episode_mem_usage)), episode_mem_usage, alpha=0.5, label='Memory Usage')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Memory Usage')
    ax5.set_title('Memory Usage Over Time')

    ax6 = ax5.twinx()
    for episode_num_vm in episode_num_vms:
        ax6.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', label='Num VMs')
    ax6.set_ylabel('Number of VMs')

    ax7 = plt.subplot(3, 2, 4)
    for episode_cpu_usage in episode_cpu_usages:
        ax7.plot(range(len(episode_cpu_usage)), episode_cpu_usage, alpha=0.5, label='CPU Usage')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('CPU Usage')
    ax7.set_title('CPU Usage Over Time')

    ax8 = ax7.twinx()
    for episode_num_vm in episode_num_vms:
        ax8.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', label='Num VMs')
    ax8.set_ylabel('Number of VMs')

    ax9 = plt.subplot(3, 2, 5)
    for episode_latency in episode_latencies:
        ax9.plot(range(len(episode_latency)), episode_latency, alpha=0.5, label='Latency')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Latency')
    ax9.set_title('Latency Over Time')

    ax10 = ax9.twinx()
    for episode_num_vm in episode_num_vms:
        ax10.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', label='Num VMs')
    ax10.set_ylabel('Number of VMs')

    ax11 = plt.subplot(3, 2, 6)
    for episode_num_vm in episode_num_vms:
        ax11.plot(range(len(episode_num_vm)), episode_num_vm, alpha=0.5, label='Number of VMs')
    ax11.set_xlabel('Time')
    ax11.set_ylabel('Number of VMs')
    ax11.set_title('Number of VMs Over Time')

    plt.tight_layout()
    plt.savefig(f'{output_filename_prefix}_metrics.jpg')
    plt.close()

if __name__ == '__main__':
    train_steps_list = [2, 5, 10, 20]
    eval_steps = 4000
    max_episodes_per_step = 1000
    rewards_summary = []

    for train_steps in train_steps_list:
        print(f"Training for {train_steps} steps...")
        episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, training_rewards = train_agent(train_steps, max_episodes_per_step)
        print("Training completed.\n")
    
    agent = SimpleAgent()

    for train_steps in train_steps_list:
        print(f"Evaluating using model trained with {train_steps} steps...")
        
        # Evaluate using evaluate_agent function
        episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, evaluation_rewards = evaluate_agent(agent, eval_steps)
        
        # Plot metrics for evaluation
        output_prefix = f'evaluation_plots_model_{train_steps}'
        plot_metrics(episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix=f"plots_{train_steps}")

        # Collect rewards for summary plot
        rewards_summary.append((train_steps, sum(evaluation_rewards)))

        print(f"Evaluation completed for model trained with {train_steps} steps.\n")

    # Plotting the summary of rewards
    plt.figure()
    train_steps, total_rewards = zip(*rewards_summary)
    plt.bar(train_steps, total_rewards, tick_label=[f'Model {ts} steps' for ts in train_steps])
    plt.xlabel('Training Steps')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards for Models Trained with Different Steps')
    plt.savefig('rewards_summary.jpg')
    plt.close()

    print("All evaluations completed.")
