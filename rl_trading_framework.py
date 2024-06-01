import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from rl_functions import *
from config import config
import time


# State encoding model (LSTM)
class StateEncodingModel(nn.Module):
    def __init__(self, device, n_of_device=torch.cuda.device_count(), input_dim=8, lstm_hidden_dim=150, num_layers=1):
        """
        :param input_dim: is 8 in the paper, 3 dim from action_t-1, 2 dim from pairs of asset prices, 2 dim for log prices, 1 dim from normalized spread of log prices
        :param lstm_hidden_dim: integer, dimension of the hidden state of LSTM
        :param num_layers: integer, number of lstm layers
        # default values all from paper, paper saying h_dim 300 is very good
        """
        super(StateEncodingModel, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.n_of_device = n_of_device

        # single-layer LSTM
        # when batch_first=True, lstm input and output dims are (batch_size, seq_len, n_feature)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers, batch_first=True)
        self.state = [None] * self.n_of_device    # one state tuple per gpu
        self.init_weights()
        # self.reset_state(batch_size_per_device)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # batch_size: pairs of assets k, seq_length=1, input_dim=8

        lstm_out, next_state = self.lstm(x)
        # lstm_out: Yt, hidden: ht and ct, in a standard LSTM, Yt = ht, dim: (batch_size, seq_len, hidden_dim)

        # Update state
        self.state = next_state

        # Taking the last output from LSTM (at time t)
        lstm_out_last = lstm_out[:, -1, :]  # dim: (batch_size, hidden_dim)

        return lstm_out_last

    def reset_state(self, batch_size_per_device):
        hidden_state = torch.zeros(self.num_layers, batch_size_per_device, self.lstm_hidden_dim).to(self.device)
        cell_state = torch.zeros(self.num_layers, batch_size_per_device, self.lstm_hidden_dim).to(self.device)
        self.state = hidden_state, cell_state


    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class TradingPolicyModel(nn.Module):
    def __init__(self, lstm_hidden_dim, mlp_hidden_dim, action_dim):
        """
        :param lstm_hidden_dim: integer, dimension of the hidden state of LSTM
        :param mlp_hidden_dim:  integer, dim of a single-layer MLP
        :param action_dim: integer, dim of action space, should be 3
        """
        super(TradingPolicyModel, self).__init__()

        # 2 MLP layers (from paper code)
        # todo: add the ability to add multiple MLP layer
        self.fc1 = nn.Linear(in_features=lstm_hidden_dim + 1, out_features=mlp_hidden_dim)  # +1 for portfolio value
        self.fc2 = nn.Linear(in_features=mlp_hidden_dim, out_features=action_dim)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, encoded_state, norm_port_value):
        # Concatenate the encoded state with the normalized portfolio value
        # encoded_state is the output of the state_encoding_model (lstm)
        concat = torch.cat([encoded_state, norm_port_value], dim=1)
        # dim of concat is (batch_size, lstm_hidden_dim + 1)

        # MLP layers
        mlp_out = F.leaky_relu(self.fc1(concat))

        # Softmax layer to get action probabilities
        action_probs = F.softmax(self.fc2(mlp_out), dim=1)

        return action_probs


class TradingAgent:
    def __init__(self, state_encoding_model, trading_policy_model, optimizer_class=optim.Adam, learning_rate=0.001):
        self.state_encoding_model = state_encoding_model
        self.trading_policy_model = trading_policy_model
        self.optimizer = optimizer_class(list(self.trading_policy_model.parameters()) +
                                         list(self.state_encoding_model.parameters()),
                                         lr=learning_rate)
        self.loss_history_all = []
        self.mean_loss_history_all = []
        self.mean_reward_all = []

    def reset(self):
        # reset for next episode
        self.log_action_probs = []      # final dim will be [t, 1], [t, k] if batch_size=k
        self.rewards = []               # final dim will be [t, 1], [t, k] if batch_size=k
        self.episode_mean_losses = None
        self.episode_mean_rewards = None

    def choose_action(self, encoded_state, norm_port_value, method="random"):
        action_probs = self.trading_policy_model.forward(encoded_state, norm_port_value)
        actions = get_action(action_probs, method)
        selected_log_probs = torch.log(action_probs[range(len(actions)), actions])
        self.log_action_probs.append(selected_log_probs)

        return actions

    def pre_learn_and_retain_pairs(self, gamma=0.99):

        # for learn
        # calculate discounted rewards, policy loss
        discounted_r = discounted_rewards_torch(self.rewards, gamma)
        # Standardize the discounted rewards for each individual reward
        mean = discounted_r.mean(dim=0, keepdim=True)
        std = discounted_r.std(dim=0, keepdim=True)
        standardized_discounted_r = (discounted_r - mean) / (std + 1e-9)
        # todo could try to use moving average of mean and std, making agent more sensitive to recent rewards
        self.log_action_probs = torch.stack(self.log_action_probs, dim=0)
        loss = -self.log_action_probs * standardized_discounted_r

        # for calculating performance score and retaining pairs
        self.episode_mean_rewards = torch.mean(discounted_r, dim=0).detach()
        self.episode_mean_losses = torch.mean(loss, dim=0).detach()
        self.mean_reward_all.append(torch.mean(self.episode_mean_rewards, dim=0).item())

        return loss
        
    def learn(self, gamma=0.99):

        loss = self.pre_learn_and_retain_pairs(gamma)
        # backward and optimize
        self.optimizer.zero_grad()
        loss_sum = loss.sum()
        loss_sum.backward()

        # Clipping gradients to prevent gradient explosion
        '''
        torch.nn.utils.clip_grad_norm_(
            list(self.trading_policy_model.parameters()) +
            list(self.state_encoding_model.parameters()),
            max_norm=1
        )
        '''
        self.optimizer.step()
        
        self.loss_history_all.append(loss_sum.item())
        batch_size = len(self.log_action_probs)
        self.mean_loss_history_all.append(loss_sum.item()/batch_size)

    def retain_pairs(self, original_pairs, alpha, beta, temperature, retain_threshold, adjust_for_n_gpu=True, n_of_gpu=None):
    # run this function after learn
        rewards_mean, rewards_std = cal_stats_torch(self.episode_mean_rewards, dim=0)
        losses_mean, losses_std = cal_stats_torch(self.episode_mean_losses, dim=0)
        standardized_rewards = (self.episode_mean_rewards - rewards_mean) / (rewards_std + 1e-9)  # Adding a small value to avoid division by zero
        standardized_losses = (self.episode_mean_losses - losses_mean) / (losses_std + 1e-9)

        performance_score = alpha * standardized_rewards - beta * standardized_losses
        retain_prob = torch.sigmoid(performance_score/temperature)
        retained_idx = torch.where(retain_prob > retain_threshold)[0]
        retained_pairs = [original_pairs[i] for i in retained_idx]

        if adjust_for_n_gpu == True:
            retained_pairs = adjust_samples_for_n_gpu(retained_pairs, n_of_gpu)

        return retained_pairs


class TradingEnvironment:
    def __init__(self, state_encoding_model, reg_rolling_window, portfolio_settings, action_dim, n_of_gpu):

        self.state_encoding_model = state_encoding_model
        self.reg_rolling_window = reg_rolling_window
        self.bt_settings = portfolio_settings
        self.action_dim = action_dim
        self.n_of_gpu = n_of_gpu

        # update when running update_data
        self.price_data = None
        self.period_len = None

        # reset every episode
        self.features_data = None
        self.pair_indices = None
        self.batch_size = None
        self.current_time_idx = None
        self.done = False
        self.cash = None
        self.port_value = None
        self.norm_port_value = None
        self.positions = None
        self.y_quantity = None
        self.x_quantity = None

    def update_data(self, price_data):
        self.price_data = price_data
        self.period_len = len(price_data)

    def reset(self, pair_indices):
        # reset the env at the beginning of a new episode
        device = self.price_data.device
        self.pair_indices = pair_indices
        self.features_data = calculate_features(self.price_data, self.reg_rolling_window, self.pair_indices)
        self.batch_size = len(self.pair_indices)
        self.current_time_idx = self.reg_rolling_window     # sample_data[0:reg_rolling_window] cannot be used
        self.done = False
        self.cash = torch.ones(self.batch_size, device=device) * self.bt_settings['initial_cash']
        self.port_value = self.cash
        self.norm_port_value = torch.ones(self.batch_size).to(device)
        self.positions = torch.zeros(self.batch_size, dtype=torch.long).to(device)     # positions is last action, not quantity
        self.y_quantity = torch.zeros(self.batch_size).to(device)
        self.x_quantity = torch.zeros(self.batch_size).to(device)

    def update_observation(self):
        # time_idx must be greater than reg_rolling_window
        # ob1 dim is (k, 3), ob2 dim is (k, 5)
        ob1 = F.one_hot(self.positions, self.action_dim)
        ob2 = self.features_data[self.current_time_idx]
        ob_t = torch.cat((ob1, ob2), dim=1).unsqueeze(1)
        # add 1 dim: seq_len, when batch_first=True in lstm, input and output dims are (batch_size, seq_len, n_feature)

        return ob_t

    def step(self, action):

        port_value_holder = self.port_value

        self.y_quantity, self.x_quantity, self.cash, self.port_value = update_quant_cash_pv_no_lvg_torch(
            price=self.features_data[self.current_time_idx][:, :2],
            current_y_quantity=self.y_quantity,
            current_x_quantity=self.x_quantity,
            cash=self.cash,
            port_value=self.port_value,
            current_position=self.positions,
            action=action
        )

        reward = self.port_value - port_value_holder
        # todo: this may encourage the agent to not trade often, can add other components to reward calculation such as
        #  min return, Sharpe ratio, or increase gamma
        self.positions = action

        # step forward
        self.current_time_idx += 1
        done = self.current_time_idx == self.period_len
        if not done:
            next_ob = self.update_observation()
        else:
            next_ob = None

        return reward, next_ob, done


def setup(rank, world_size):
    # for pytorch distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, price_data_all, pair_indices, **kwargs):
    cfg = dict(config)
    cfg.update(kwargs)  # equivalent to cfg = {key: kwargs.get(key, config[key]) for key in config}
   
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    writer = SummaryWriter(f'runs/training_process_{rank}')
    start_t = time.time()

    state_encoding_model = StateEncodingModel(device, input_dim=8, lstm_hidden_dim=cfg['lstm_hidden_dim'], num_layers=cfg['lstm_num_layers']).to(device)
    trading_policy_model = TradingPolicyModel(cfg['lstm_hidden_dim'], cfg['mlp_hidden_dim'], cfg['action_dim']).to(device)
    trading_agent = TradingAgent(state_encoding_model, trading_policy_model, cfg['optimizer_class'], cfg['learning_rate'])
    env = TradingEnvironment(state_encoding_model, cfg['reg_rolling_window'], cfg['portfolio_settings'], cfg['action_dim'], cfg['n_of_gpu'])
    state_encoding_model = DDP(state_encoding_model, device_ids=[rank])
    trading_policy_model = DDP(trading_policy_model, device_ids=[rank])

    total_rolling = len(range(0, price_data_all.shape[0] - cfg['rolling_window_size'], cfg['rolling_step_size']))
    total_episodes = total_rolling * cfg['episodes_per_rolling']
    if rank == 0:
        model_folder_path = create_model_folder(cfg, check_exists=True, check_version=True)
        print('Total rolling:', total_rolling)
        print('Total episodes:', total_episodes)
        print('')
    else:
        model_folder_path = create_model_folder(cfg, check_exists=False, check_version=False)

    price_data_all = price_data_all.cuda(rank)
    retained_pairs = pair_indices
    retained_pairs_len = []
    retain_threshold = cfg['initial_retain_threshold']
    temperature = cfg['initial_T']
    drop_start = int(total_episodes * cfg['drop_start_ratio'])
    drop_end = int(total_episodes * cfg['drop_end_ratio'])
    next_drop = drop_start
    episode_count = 0
    model_saved_at = 0
    best_loss = float('inf')

    pbar = tqdm(total=total_episodes, desc="Training Progress")

    for start_idx in range(0, price_data_all.shape[0]-cfg['rolling_window_size'], cfg['rolling_step_size']):
        
        price_data = price_data_all[start_idx:start_idx+cfg['rolling_window_size'], :]
        env.update_data(price_data)
        
        for _ in range(cfg['episodes_per_rolling']):

            episode_count += 1
            # todo: check epsilon-greedy
            if episode_count <= total_episodes * cfg['exploration_fraction']:
                method = 'random'
            else:
                method = 'greedy'
                
    #         temperature = initial_T * (T_decay_rate ** episode_count)
    #         retain_threshold = initial_retain_threshold * (retain_threshold_growth_rate ** episode_count)
    #         retain_threshold = initial_retain_threshold + 0.01 * episode_count

            retained_pairs = filter_pairs_with_nan(price_data, pair_indices)
            batch_size_train = len(retained_pairs)
            retained_pairs_len.append(batch_size_train)
            
            # 1. reset model and env, initiate observation
            if isinstance(state_encoding_model, DDP):
                state_encoding_model.module.reset_state(batch_size_train)
            else:
                state_encoding_model.reset_state(batch_size_train)
            trading_agent.reset()
            env.reset(pair_indices=retained_pairs)
            ob_t = env.update_observation()
            done = False
            trajectory = []
            
            while not done:
                # 2. lstm decides a state

                encoded_state = state_encoding_model(ob_t)
                env.norm_port_value = (env.port_value / env.bt_settings['initial_cash']).unsqueeze(1)   # unsqueeze dim from (batch_size_train) to (batch_size_train, 1) for torch.cat

                # 3. agent makes a decision
                action = trading_agent.choose_action(encoded_state, env.norm_port_value, method)    # action is the desired position at time t

                # 4. calculate reward, update observations based on action
                reward, ob_t, done = env.step(action)
                trading_agent.rewards.append(reward)

                # 5. store state, action, reward
                trajectory.append((encoded_state, action, reward))
                
            # 6. learn after getting the whole trajectory
            trading_agent.rewards = torch.stack(trading_agent.rewards)
            trading_agent.learn(gamma=0.99)

            current_loss = trading_agent.loss_history_all[-1]
            current_mean_loss = trading_agent.mean_loss_history_all[-1]
            current_mean_reward = trading_agent.mean_reward_all[-1]
            writer.add_scalar('Total Loss', current_loss, episode_count)
            writer.add_scalar('Mean Loss', current_mean_loss, episode_count)
            writer.add_scalar('Mean Reward', current_mean_reward, episode_count)
            
            # 7. filter pairs
            if drop_start <= episode_count <= drop_end and episode_count >= next_drop and len(retained_pairs) > cfg['stop_drop_threshold']:
                
                retained_pairs = trading_agent.retain_pairs(retained_pairs, cfg['alpha'], cfg['beta'], temperature, retain_threshold, adjust_for_n_gpu=True, n_of_gpu=cfg['n_of_gpu'])
                next_drop = episode_count + int(total_episodes * cfg['drop_pairs_percentage_interval'])
                temperature -= 0.2
                retain_threshold += 0.08
    #             episodes_per_rolling += 10
                if rank == 0:
                    print('drop at episode:', episode_count, 'retained pairs after drop:', len(retained_pairs), 'temperature:', temperature, 'retain_threshold:', retain_threshold)
    
            # 8. save models
            if current_loss < best_loss:
                best_loss = current_loss
                model_saved_at = episode_count
                if rank == 0:
                    save_model(state_encoding_model, model_folder_path, 'state_encoding_model')
                    save_model(trading_policy_model, model_folder_path, 'trading_policy_model')

            pbar.update(1)
    
    save_training_plot_data_to_csv(trading_agent.loss_history_all, trading_agent.mean_loss_history_all, trading_agent.mean_reward_all, model_folder_path, rank)
    if rank == 0:
        save_config(cfg, model_folder_path)
        print('model saved at episode ' + str (model_saved_at) + ', configuration and training plot data saved')
        
    pbar.close()
    end_t = time.time()
    if rank == 0:
        print('Total training time:', round(end_t - start_t, 2), 'seconds')
    writer.close()

    cleanup()

