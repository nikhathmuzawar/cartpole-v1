import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(is_training=True, render=False):

    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # arrays for velocity, position, angular velocity and angle
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if is_training:
        q = np.zeros((len(pos_space) + 1, len(vel_space) + 1, len(ang_space) + 1, len(ang_vel_space) + 1, env.action_space.n))
    else:
        with open('cartpole1.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.1 
    discount_factor_g = 0.99

    epsilon = 1        
    epsilon_decay_rate = 0.00001 
    rng = np.random.default_rng() 

    rewards_per_episode = []
    losses_per_episode = [] 

    i = 0

    while True:

        state = env.reset()[0]      
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False          
        rewards = 0
        losses = [] 

        while not terminated and rewards < 10000:

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                old_value = q[state_p, state_v, state_a, state_av, action]
                next_max = np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
                td_error = reward + discount_factor_g * next_max - old_value
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * td_error
                losses.append(td_error)  

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

            if not is_training and rewards % 100 == 0:
                print(f'Episode: {i}  Rewards: {rewards}')

        rewards_per_episode.append(rewards)
        losses_per_episode.append(np.mean(losses)) 

        mean_rewards = np.mean(rewards_per_episode[-100:])

        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        i += 1

    env.close()

    # Save Q table to file
    if is_training:
        with open('cartpole1.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Plotting
    episodes = np.arange(len(rewards_per_episode))

    # Rewards per episode
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(episodes, rewards_per_episode, label='Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Rewards vs Episode')
    plt.legend()

    # Loss per episode
    plt.subplot(3, 1, 2)
    plt.plot(episodes, losses_per_episode, label='Loss per Episode', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss (TD Error) vs Episode')
    plt.legend()

    # Mean rewards
    mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(len(rewards_per_episode))]
    plt.subplot(3, 1, 3)
    plt.plot(episodes, mean_rewards, label='Mean Reward (last 100 episodes)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward vs Episode')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cartpole_metrics.png')
    plt.show()

if __name__ == '__main__':
    run(is_training=True, render=False)
    #run(is_training=False, render=True)
