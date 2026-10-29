import numpy as np
import pandas as pd
# Define the environment
from environment import Environment
from ddqn import DDQNAgent
import matplotlib.pyplot as plt
import torch



def train_agent(train_steps, max_episodes_per_step=1000):
    env = Environment()
    state_size = 6
    action_size = 3
    sequence_length = 10
    batch_size = 16
    agent = DDQNAgent(state_size, action_size, sequence_length)

    rewards = []
    episode_loads = []
    episode_read_percentages = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies=[]
    episode_num_vms = []

    for _ in range(train_steps):
        state = np.zeros((sequence_length, state_size))
        total_reward = 0
        no_improvement_counter = 0
        best_reward = -np.inf


        episode_load = []
        episode_read_percentage = []
        episode_mem_usage = []
        episode_cpu_usage = []
        episode_latency=[]
        episode_num_vm = []
        
        done = False  # Initialize done outside the loop
        step_count = 0  # Initialize step_count
        from tqdm import tqdm
        while not done:
            for episode_idx in tqdm(range(max_episodes_per_step)):
                action = agent.act(state, verbose=0)

                new_state, reward, done = env.step(action)
                total_reward += reward
                # print("here")
                # print(new_state)
                # print("Shape of next_state:", new_state.shape)

                # next_state = np.zeros((sequence_length, state_size))
                # next_state[:-1] = state[1:]
                # next_state[-1][0] = env.load(env.time + sequence_length)
                # next_state[-1][1] = env.read_percentage(env.time + sequence_length)
                # next_state[-1][2] = env.cpu_usage(env.time + sequence_length)
                # next_state[-1][3] = env.memory_usage(env.time + sequence_length)
                # next_state[-1][4] = env.latency(env.time + sequence_length)
                
                # print(next_state)
                
                
                next_state = np.zeros((sequence_length, state_size))
                next_state[:-1] = state[1:]
                next_state[-1][0] = new_state[0]
                next_state[-1][1] = new_state[1]
                next_state[-1][2] = new_state[2]
                next_state[-1][3] = new_state[3]
                next_state[-1][4] = new_state[4]
                next_state[-1][5] = new_state[5]

                #print(next_state)
                agent.remember(state, action, reward, next_state, done)

                episode_load.append(next_state[-1][0])
                episode_read_percentage.append(next_state[-1][1])
                episode_mem_usage.append(next_state[-1][2])
                episode_cpu_usage.append(next_state[-1][3])
                episode_latency.append(next_state[-1][4])

                episode_num_vm.append(next_state[-1][5])

                state = next_state

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                # if total_reward > best_reward:
                #     best_reward = total_reward
                #     no_improvement_counter = 0
                # else:
                #     no_improvement_counter += 1

                if no_improvement_counter > 100:  # Terminate if no improvement in 100 consecutive episodes
                    done = True
                    break  # Exit the loop when termination condition is met
                step_count += 1
                if step_count >= 1000:  # Terminate after 2000 steps
                    print("done")
                    done = True
                    break  # Exit the loop when termination condition is met

                if episode_idx % 2 == 0:
                    agent.plot_loss_history()

                if episode_idx % 10 == 0:
                    evaluate_agent(agent, 100)

            rewards.append(total_reward)
            episode_loads.append(episode_load)
            episode_read_percentages.append(episode_read_percentage)
            episode_mem_usages.append(episode_mem_usage)
            episode_cpu_usages.append(episode_cpu_usage)
            episode_latencies.append(episode_latency)
            episode_num_vms.append(episode_num_vm)
            if done:
                break  # Exit the outer loop when termination condition is met


    with open(f"rewards_{train_steps}.txt", "w") as f:
        for reward in rewards:
            f.write(str(reward) + "\n")

    agent.save_model(f'model_{train_steps}.pth')
    # print(f"Model for {train_steps} training steps saved to 'model_{train_steps}.h5'.")
    # print("Training completed.\n")
    return episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms


def evaluate_agent(agent, eval_steps):
    env = Environment()
    state_size = 6  # Adjusted state size to include latency
    action_size = 3
    sequence_length = 10
    batch_size = 16
    # agent = DDQNAgent(state_size, action_size, sequence_length)

    # Load the trained model weights
    # agent.online_model.load_weights(f'model_{train_steps}.weights.h5')

    total_rewards = []
    episode_loads = []
    episode_read_percentages = []
    episode_mem_usages = []
    episode_cpu_usages = []
    episode_latencies = []  # Added to store latency values
    episode_num_vms = []

    for _ in range(1):
        state = np.zeros((sequence_length, state_size))
        total_reward = 0

        episode_load = []
        episode_read_percentage = []
        episode_mem_usage = []
        episode_cpu_usage = []
        episode_latency = []  # Added to store latency values
        episode_num_vm = []

        done = False
        idx = 0
        while not done:
            idx += 1
            action = agent.act(state, verbose=0)

            new_state, reward, done = env.step(action)
            total_reward += reward

            next_state = np.zeros((sequence_length, state_size))
            next_state[:-1] = state[1:]
            next_state[-1][0] = new_state[0]
            next_state[-1][1] = new_state[1]
            next_state[-1][2] = new_state[2]
            next_state[-1][3] = new_state[3]
            next_state[-1][4] = new_state[4]
            next_state[-1][5] = new_state[5]

            #print(next_state)
            agent.remember(state, action, reward, next_state, done)

            episode_load.append(next_state[-1][0])
            episode_read_percentage.append(next_state[-1][1])
            episode_mem_usage.append(next_state[-1][2])
            episode_cpu_usage.append(next_state[-1][3])
            episode_latency.append(next_state[-1][4])

            episode_num_vm.append(next_state[-1][5])

            state = next_state

            if idx > eval_steps:
                break

        total_rewards.append(total_reward)
        episode_loads.append(episode_load)
        episode_read_percentages.append(episode_read_percentage)
        episode_mem_usages.append(episode_mem_usage)
        episode_cpu_usages.append(episode_cpu_usage)
        episode_latencies.append(episode_latency)  # Append latency episodes
        episode_num_vms.append(episode_num_vm)

    avg_reward = np.mean(total_rewards)
    print(f"Avg. reward for model trained with {train_steps} steps: {avg_reward}")
    print(f"Epsilon is {agent.epsilon}")

    
    return episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms


import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def plot_metrics(episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix):
    # Plotting metrics
    plt.figure(figsize=(12, 10))

    # Plot Load Over Time
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

    # Plot Read Percentage Over Time
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

    # Plot Memory Usage Over Time
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

    # Plot CPU Usage Over Time
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

    # Plot Latency Over Time
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

    # Plot Number of VMs Over Time
    ax11 = plt.subplot(3, 2, 6)
    for episode_num_vm in episode_num_vms:
        ax11.plot(range(len(episode_num_vm)), episode_num_vm, alpha=0.5, label='Number of VMs')
    ax11.set_xlabel('Time')
    ax11.set_ylabel('Number of VMs')
    ax11.set_title('Number of VMs Over Time')

    plt.tight_layout()
    plt.savefig(f'{output_filename_prefix}_metrics.jpg')
    plt.show()

def plot_dots(plt, episode_num_vms, label):
    for episode_num_vm in episode_num_vms:
        plt.scatter(range(len(episode_num_vm)), episode_num_vm, color='red', marker='o', label=label)



# Example usage:
# Assuming episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms are defined.
# output_prefix = 'evaluation_plots_model_10'
# plot_metrics(episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_prefix)


    # Optional: Uncomment to display the plots inline
    # plt.tight_layout()
    # plt.show()




if __name__ == '__main__':
    train_steps_list = [2, 5, 10, 50]
    eval_steps = 4000
    max_episodes_per_step = 1000

    # for train_steps in train_steps_list:
    #     print(f"Training for {train_steps} steps...")
    #     episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms = train_agent(train_steps, max_episodes_per_step)
    #     print("Training completed.\n")

    for train_steps in train_steps_list:
        print(f"Evaluating using model trained with {train_steps} steps...")
        
        # Initialize DDQNAgent
        agent = DDQNAgent(state_size=6, action_size=3, sequence_length=10)
        
        # Load corresponding model weights
        model_filename = f'model_{train_steps}.pth'
        agent.online_model.load_state_dict(torch.load(model_filename))

        # Evaluate using evaluate_agent function
        episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms = evaluate_agent(agent, eval_steps)
        
        # Plot metrics for evaluation
        output_prefix = f'evaluation_plots_model_{train_steps}'
        #plot_metrics(episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix="plots_double_y")
        plot_metrics(episode_loads, episode_read_percentages, episode_mem_usages, episode_cpu_usages, episode_latencies, episode_num_vms, output_filename_prefix=f"plots_double_y_{train_steps}")

        print(f"Evaluation completed for model trained with {train_steps} steps.\n")

    print("All evaluations completed.")



