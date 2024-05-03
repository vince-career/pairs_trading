from rl_trading_framework import *
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *

price_data_all = get_data_torch(tickers, train_start, train_end, multiplier, data_freq, device)
data_chunks = [price_data_all[i:i+time_steps_per_chunk, :] for i in range(0, price_data_all.shape[0], time_steps_per_chunk)]

state_encoding_model = StateEncodingModel(device, batch_size_train, input_dim=8, lstm_hidden_dim=lstm_hidden_dim, num_layers=lstm_num_layers).to(device)
trading_policy_model = TradingPolicyModel(lstm_hidden_dim, mlp_hidden_dim, action_dim).to(device)
trading_agent = TradingAgent(state_encoding_model, trading_policy_model, optimizer_class, learning_rate)
env = TradingEnvironment(state_encoding_model, reg_rolling_window, batch_size_train, portfolio_settings, action_dim, device, full_sample_train, pair_indices_given=[])

plt.ion()  # Turn on interactive mode
total_episodes = episodes_per_chunk * len(data_chunks)
episode_count = 0
start_t = time.time()
for i, price_data in enumerate(data_chunks):
    env.update_data(price_data)
    for episode in tqdm(range(episodes_per_chunk), desc='training ' + str(i+1) + '/' + str(len(data_chunks))):
        # 1. reset model and env, initiate observation
        state_encoding_model.reset_state(batch_size_train)
        trading_agent.reset()
        # re_t = time.time()
        env.reset()
        # print('reset env used', time.time()-re_t)
        ob_t = env.update_observation()
        done = False
        trajectory = []

        while not done:
            # 2. lstm decides a state
            encoded_state = state_encoding_model(ob_t)
            env.norm_port_value = (env.port_value / env.bt_settings['initial_cash']).unsqueeze(1)
            # change dim from (batch_size_train) to (batch_size_train, 1) for torch.cat

            # 3. agent makes a decision
            # todo check epsilon-greedy
            if episode_count <= total_episodes * exploration_fraction:
                method = 'random'
            else:
                method = 'greedy'
            action = trading_agent.choose_action(encoded_state, env.norm_port_value, method)
            # action is the desired position at time t

            # 4. calculate reward, update observations based on action
            reward, ob_t, done = env.step(action)
            trading_agent.rewards.append(reward)

            # 5. store state, action, reward
            trajectory.append((encoded_state, action, reward))
            episode_count += 1

        # 6. learn after getting the whole trajectory
        trading_agent.rewards = torch.stack(trading_agent.rewards)
        trading_agent.learn(gamma=0.99)
        trading_agent.plot_loss()

        trading_agent.plot_loss()

end_t = time.time()
print('Total training time:', round(end_t - start_t, 2), 'seconds')

# save models
save_model(state_encoding_model, model_folder_path, encoding_model_file_name)
save_model(trading_policy_model, model_folder_path, policy_model_file_name)

plt.ioff()      # in trading_agent.plot_loss(), plt.ion()
plt.show()
