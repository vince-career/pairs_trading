import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl_functions import *
import matplotlib.pyplot as plt


# State encoding model (LSTM)
class StateEncodingModel(nn.Module):
    def __init__(self, device, batch_size, input_dim=8, lstm_hidden_dim=150, num_layers=1):
        """
        :param input_dim: should be 8, 3 dim from action_t-1, 2 dim from pairs of asset prices, 2 dim for log prices, 1 dim from normalized spread of log prices
        :param lstm_hidden_dim: integer, dimension of the hidden state of LSTM
        :param num_layers: integer, number of lstm layers
        # default values all from paper, paper saying h_dim 300 is very good
        """
        super(StateEncodingModel, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.device = device

        # single-layer LSTM
        # when batch_first=True, lstm input and output dims are (batch_size, seq_len, n_feature)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers, batch_first=True)
        self.state = None, None
        self.init_weights()
        self.reset_state(batch_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # batch_size: pairs of assets k, seq_length=1, input_dim=8
        lstm_out, self.state = self.lstm(x, self.state)
        # lstm_out: Yt, hidden: ht and ct, in a standard LSTM, Yt = ht, dim: (batch_size, seq_len, hidden_dim)

        # Taking the last output from LSTM (at time t)
        lstm_out_last = lstm_out[:, -1, :]  # dim: (batch_size, hidden_dim)

        return lstm_out_last

    def reset_state(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to(self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to(self.device)
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
        self.log_action_probs = []      # final dim will be [t, 1], [t, k] if batch_size=k
        self.rewards = []               # final dim will be [t, 1], [t, k] if batch_size=k
        self.loss_history = []

    def reset(self):
        # reset for next episode
        self.log_action_probs = []
        self.rewards = []

    def choose_action(self, encoded_state, norm_port_value, method="random"):
        action_probs = self.trading_policy_model.forward(encoded_state, norm_port_value)
        actions = get_action(action_probs, method)
        # print(actions)
        selected_log_probs = torch.log(action_probs[range(len(actions)), actions])
        self.log_action_probs.append(selected_log_probs)

        return actions

    def learn(self, gamma=0.99):

        # calculate discounted rewards, policy loss
        discounted_r = discounted_rewards_torch(self.rewards, gamma)

        # Normalize the discounted rewards for each individual reward
        mean = discounted_r.mean(dim=0, keepdim=True)
        std = discounted_r.std(dim=0, keepdim=True)
        norm_discounted_r = (discounted_r - mean) / (std + 1e-9)
        # todo could try to use moving average of mean and std, making agent more sensitive to recent rewards
        self.log_action_probs = torch.stack(self.log_action_probs, dim=0)
        loss = -self.log_action_probs * norm_discounted_r

        # backward and optimize
        self.optimizer.zero_grad()
        loss_sum = loss.sum()
        loss_sum.backward()

        # Clipping gradients to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(
            list(self.trading_policy_model.parameters()) +
            list(self.state_encoding_model.parameters()),
            max_norm=1
        )

        self.optimizer.step()
        self.loss_history.append(loss_sum.item())

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Training Loss over Time")
        plt.xlabel("Episode Step")
        plt.ylabel("Loss")
        plt.draw()
        plt.pause(0.1)  # Adjust the pause duration


class TradingEnvironment:
    def __init__(self, state_encoding_model, reg_rolling_window, batch_size, portfolio_settings, action_dim, device, full_sample, pair_indices_given):

        self.state_encoding_model = state_encoding_model
        self.batch_size = batch_size
        self.reg_rolling_window = reg_rolling_window
        self.bt_settings = portfolio_settings
        self.action_dim = action_dim
        self.device = device
        self.full_sample = full_sample
        self.pair_indices_given = pair_indices_given

        self.price_data = None
        self.period_len = None

        # reset every episode
        self.features_data = None
        self.pair_indices = None
        self.current_time_idx = None
        self.done = False
        self.cash = None
        self.port_value = None
        self.norm_port_value = None
        self.positions = None
        self.x_quantity = None
        self.y_quantity = None

    def update_data(self, price_data):
        self.price_data = price_data
        self.period_len = len(price_data)

    def reset(self):
        # reset the env at the beginning of a new episode
        self.features_data, self.pair_indices = generate_pairs_and_features_torch(self.price_data, self.batch_size, self.reg_rolling_window, self.device, self.full_sample, self.pair_indices_given)
        self.current_time_idx = self.reg_rolling_window     # sample_data[0:reg_rolling_window] cannot be used
        self.done = False
        self.cash = torch.ones(self.batch_size, device=self.device) * self.bt_settings['initial_cash']
        self.port_value = self.cash
        self.norm_port_value = torch.ones(self.batch_size).to(self.device)
        self.positions = torch.zeros(self.batch_size, dtype=torch.long).to(self.device)     # positions is last action, not quantity
        self.x_quantity = torch.zeros(self.batch_size).to(self.device)
        self.y_quantity = torch.zeros(self.batch_size).to(self.device)

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

        self.x_quantity, self.y_quantity, self.cash, self.port_value = update_quant_cash_pv_no_lvg_torch(
            price=self.features_data[self.current_time_idx][:, :2],
            current_x_quantity=self.x_quantity,
            current_y_quantity=self.y_quantity,
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


