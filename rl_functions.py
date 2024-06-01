import numpy as np
import torch
from polygon_v9 import Stock
from tqdm import tqdm
import pandas as pd
import functools as ft
import os
import matplotlib.pyplot as plt
from datetime import datetime
import random
from collections import OrderedDict
import json
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def cal_stats_torch(sample, dim):
    return torch.mean(sample, dim=dim), torch.std(sample, dim=dim)


def get_regression_params_torch(x, y, device):
    # adjust x and y to common dimension (batch size, sample size)
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)

    batch_size, sample_size = x.shape
    ones = torch.ones(batch_size, sample_size, dtype=x.dtype, device=device)
    X = torch.stack([ones, x], dim=-1)

    # solve using Normal Equation, X^T * X * beta = X^T Y
    XtX = torch.bmm(X.transpose(1, 2), X)
    XtY = torch.bmm(X.transpose(1, 2), y.unsqueeze(-1))
    betas = torch.linalg.solve(XtX, XtY).squeeze(-1)
    intercepts, slopes = betas[:, 0], betas[:, 1]

    return intercepts, slopes


def get_regression_params_batch(x, y, device):
    # Adjust x and y to common dimension (batch size, sample size)
    x = x.transpose(0, 1).unsqueeze(-1)  # Shape (batch_size, sample_size, 1)
    y = y.transpose(0, 1).unsqueeze(-1)  # Shape (batch_size, sample_size, 1)
    
    # Prepend ones to x for the intercept term
    ones = torch.ones_like(x, dtype=x.dtype, device=device)
    X = torch.cat([ones, x], dim=-1)  # Shape (batch_size, sample_size, 2)

    # Use torch.linalg.lstsq to solve the least squares problem
    solution = torch.linalg.lstsq(X, y).solution
    
    # Extract intercepts and slopes
    intercepts = solution[:, 0, 0]  # The first coefficient is the intercept
    slopes = solution[:, 1, 0]      # The second coefficient is the slope

    return intercepts, slopes


def adjust_samples_for_n_gpu(samples, n_of_gpu):
    remainder = len(samples) % n_of_gpu
    additional_samples_needed = 0 if remainder == 0 else n_of_gpu - remainder
    additional_samples = random.choices(samples, k=additional_samples_needed)
    adjusted_samples = samples + additional_samples

    return adjusted_samples


def generate_pairs(all_pairs, n_of_gpu, full_sample=None, k=None, pair_indices_given=None):
    """
    three models:
    1). full_sample = True, returns all possible pairs from all combination
    2). agent evaluates specific pair indices
    3). randomly returns k pairs from all combination
    """

    if full_sample:     # model 1)
        if bool(pair_indices_given): # and (len(pair_indices_given) != len(all_pairs)):
            raise ValueError('If you choose full_sample=True, do not define pair_indices_given')
        if k is not None:
            raise ValueError('If you choose full_sample=True, do not define k')
        # k = len(all_pairs)
        pair_indices = all_pairs
        
    else:
        if bool(pair_indices_given):    # model 2)
            if k is not None:
                raise ValueError("k means randomly select k pairs, if you pass pair_indices_given into the function, do not define k")
            else:
                # k = len(pair_indices_given)
                pair_indices = pair_indices_given
        
        if isinstance(k, int) and k > 0:      # model 3)
            sampled = torch.randperm(len(all_pairs))[:k]
            pair_indices = [all_pairs[i] for i in sampled]
        elif k is not None:
            raise ValueError('k must be a positive integer')

    pair_indices = adjust_samples_for_n_gpu(pair_indices, n_of_gpu)

    return pair_indices


def calculate_features(price_data, rolling_window, pair_indices):
    """
    :param price_data: tensor with dim (t, n), t: timestamp, n: number of asset prices
    :param all_pairs: a list of all possible pair indices
    :param rolling_window: integer, window to calculate normalized log price and spread
    :param n_of_gpu: integer, number of GPUs
    :param full_sample: boolean, flag to decide whether to sample all combinations or not
    :param k: integer, randomly select k pairs
    :param pair_indices_given: list of tuples, indices of the price_data the agent is going to use/evaluate,
    :return: numpy array with dim (t, k, 5)
    :return: numpy array with dim (t, k, 9)

    # features:
    for a specific t and k, the first 2 dim are the asset prices of a pair,
    the 3rd and 4th dim are normalized log prices for rolling_window
    the 5th to 8th dim are mean1, std1, mean2, std2 of log prices
    the last dim is the spread of the normalized log price estimated by rolling window regression.
    # spread Zt = log(price1) - slope * log(price2) - intercept
    """

    t, n = price_data.shape
    device = price_data.device
    k = len(pair_indices)

    log_prices = torch.log(price_data)
    result = torch.zeros((t, k, 5), device=device)

    price1_idx = [pair[0] for pair in pair_indices]
    price2_idx = [pair[1] for pair in pair_indices]

    price1 = price_data[:, price1_idx]
    price2 = price_data[:, price2_idx]

    log_price1 = log_prices[:, price1_idx]
    log_price2 = log_prices[:, price2_idx]

    result[:, :, 0] = price1
    result[:, :, 1] = price2

    for i in range(rolling_window, t):
        sample1 = log_price1[i-rolling_window:i, :]
        sample2 = log_price2[i-rolling_window:i, :]

        mean1, std1 = cal_stats_torch(sample1, dim=0)
        mean2, std2 = cal_stats_torch(sample2, dim=0)

        norm_log_price1 = (log_price1[i] - mean1) / std1
        norm_log_price2 = (log_price2[i] - mean2) / std2

        intercepts, slopes = get_regression_params_batch(y=sample1, x=sample2, device=device)
        spread = log_price1[i, :] - (slopes * log_price2[i, :] + intercepts)    # residual, observed y - predicted y 

        result[i, :, 2] = norm_log_price1
        result[i, :, 3] = norm_log_price2
        result[i, :, 4] = spread

        # result[i, :, 2] = mean1
        # result[i, :, 3] = std1
        # result[i, :, 4] = mean2
        # result[i, :, 5] = std2

    return result


def split_pair_indices(pair_indices_all, rank, world_size):
    """
    Splits pair_indices_all into world_size subsets,
    and each process gets one of the subsets based on its rank.

    :param pair_indices_all: A list of tuples containing all pairs.
    :param rank: The rank of the current process (starting from 0).
    :param world_size: The total number of processes (number of GPUs).
    :return: The subset of pair_indices assigned to the current process.
    """
    assert len(pair_indices_all) % world_size == 0, "the length of pair_indices_all cannot divided by world_size"

    pairs_per_process = len(pair_indices_all) // world_size
    start_index = rank * pairs_per_process
    end_index = start_index + pairs_per_process

    return pair_indices_all[start_index:end_index]



def get_action(action_probs, method="random"):
    """
    Select an action based on the given method.

    Parameters:
    - action_probs: 2D tensor where each row is the action prob vector for a set of actions.
    - method: String indicating the selection method - either "greedy" or "random".
    at the beginning of training, better use random to explore better strategies

    Returns:
    - selected_action: The index of the selected action.
    """

    if method == "greedy":
        selected_action = torch.argmax(action_probs, dim=1)
    elif method == "random":
        selected_action = torch.cat([torch.multinomial(probs, 1) for probs in action_probs], dim=0)
    else:
        raise ValueError("Invalid method. Choose either 'greedy' or 'random'.")

    return selected_action


def discounted_rewards_torch(rewards, gamma=0.99):
    t_steps = torch.arange(rewards.size(0)).reshape(-1, 1).float().to(rewards.device)
    r = rewards * gamma**t_steps
    discounted_r = r.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]) / gamma**t_steps

    return discounted_r


def update_quant_cash_pv_no_lvg_torch(price, current_y_quantity, current_x_quantity, cash, port_value, current_position, action):
    # for optimizing the model and finding trading signal only, cannot be used in backtesting directly
    # model assumptions: no leverage: long_margin = 1, short_margin = 1, tx_cost = 0, no bid-ask spread
    # effectively, the whole portfolio will be split into two equal small portfolios for trading X and Y separately
    # some definitions: spread: Y-X, action=0: no position, action=1: long Y short X (lysx), action=2: short Y long X (sylx)

    long_margin = 1     # if changed, buying_power and shorting_power also need to be changed
    short_margin = 1
    y_price = price[:, 0]
    x_price = price[:, 1]

    # close positions if necessary
    equity_value_if_current_lysx = current_y_quantity * y_price * long_margin + current_x_quantity * x_price * short_margin
    equity_value_if_current_sylx = current_y_quantity * y_price * short_margin + current_x_quantity * x_price * long_margin
    cash_if_close = cash + torch.where(current_position == 1, equity_value_if_current_lysx, torch.where(current_position == 2, equity_value_if_current_sylx, 0))

    need_to_change = action != current_position
    cash_after_close = torch.where(need_to_change, cash_if_close, cash)
    pv_after_close = torch.where(need_to_change, cash_if_close, port_value)     # if close, pv = cash

    # open new positions
    buying_power = cash_after_close * (1/2) / long_margin
    shorting_power = cash_after_close * (1/2) / short_margin

    y_quantity_long = (buying_power / y_price).floor()
    y_quantity_short = -(shorting_power / y_price).floor()
    x_quantity_long = (buying_power / x_price).floor()
    x_quantity_short = -(shorting_power / x_price).floor()

    lysx_cash = cash_after_close - y_quantity_long * y_price * long_margin - x_quantity_short * x_price * short_margin
    sylx_cash = cash_after_close - y_quantity_short * y_price * short_margin - x_quantity_long * x_price * long_margin

    new_y_quantity = torch.where(need_to_change, torch.where(action == 1, y_quantity_long, torch.where(action == 2, y_quantity_short, 0)), current_y_quantity)
    new_x_quantity = torch.where(need_to_change, torch.where(action == 1, x_quantity_short, torch.where(action == 2, x_quantity_long, 0)), current_x_quantity)
    new_cash = torch.where(need_to_change, torch.where(action == 1, lysx_cash, torch.where(action == 2, sylx_cash, cash_after_close)), cash)

    lysx_pv = torch.where(need_to_change, lysx_cash, cash) + new_y_quantity * y_price * long_margin + new_x_quantity * x_price * short_margin
    sylx_pv = torch.where(need_to_change, sylx_cash, cash) + new_y_quantity * y_price * short_margin + new_x_quantity * x_price * long_margin
    new_port_value = torch.where(action == 1, lysx_pv, torch.where(action == 2, sylx_pv, pv_after_close))

    return new_y_quantity, new_x_quantity, new_cash, new_port_value


def update_quant_cash_pv_2x_short_torch(price, current_x_quantity, current_y_quantity, cash, port_value, current_position, action):
    # all in pairs trading
    # model assumptions: long_margin = 1, short_margin = 0.5, tx_cost = 0, no bid-ask spread
    # Value Neutral: buying power = 2/3 * cash, using 2/3 * cash, shorting power = 2/3 * cash, using 1/3 * cash

    long_margin = 1     # if changed, buying_power and shorting_power also need to be changed
    short_margin = 0.5
    y_price = price[:, 0]
    x_price = price[:, 1]

    # close positions if necessary
    equity_value_if_current_lysx = current_y_quantity * y_price * long_margin + current_x_quantity * x_price * short_margin
    equity_value_if_current_sylx = current_y_quantity * y_price * short_margin + current_x_quantity * x_price * long_margin
    cash_if_close = cash + torch.where(current_position == 1, equity_value_if_current_lysx, torch.where(current_position == 2, equity_value_if_current_sylx, 0))

    need_to_change = action != current_position
    cash_after_close = torch.where(need_to_change, cash_if_close, cash)
    pv_after_close = torch.where(need_to_change, cash_if_close, port_value)     # if close, pv = cash

    # open new positions
    buying_power = cash_after_close * (2/3) / long_margin
    shorting_power = cash_after_close * (1/3) / short_margin

    y_quantity_long = (buying_power / y_price).floor()
    y_quantity_short = -(shorting_power / y_price).floor()
    x_quantity_long = (buying_power / x_price).floor()
    x_quantity_short = -(shorting_power / x_price).floor()

    lysx_cash = cash_after_close - y_quantity_long * y_price * long_margin - x_quantity_short * x_price * short_margin
    sylx_cash = cash_after_close - y_quantity_short * y_price * short_margin - x_quantity_long * x_price * long_margin

    new_y_quantity = torch.where(need_to_change, torch.where(action == 1, y_quantity_long, torch.where(action == 2, y_quantity_short, 0)), current_y_quantity)
    new_x_quantity = torch.where(need_to_change, torch.where(action == 1, x_quantity_short, torch.where(action == 2, x_quantity_long, 0)), current_x_quantity)
    new_cash = torch.where(need_to_change, torch.where(action == 1, lysx_cash, torch.where(action == 2, sylx_cash, cash_after_close)), cash)

    lysx_pv = torch.where(need_to_change, lysx_cash, cash) + new_y_quantity * y_price * long_margin + new_x_quantity * x_price * short_margin
    sylx_pv = torch.where(need_to_change, sylx_cash, cash) + new_y_quantity * y_price * short_margin + new_x_quantity * x_price * long_margin
    new_port_value = torch.where(action == 1, lysx_pv, torch.where(action == 2, sylx_pv, pv_after_close))

    return new_y_quantity, new_x_quantity, new_cash, new_port_value


def check_parameters(model):
    for name, param in model.named_parameters():
        print(name, param)
        if torch.isnan(param).any():
            print(f"Parameter {name} has NaN values!")
        if torch.isinf(param).any():
            print(f"Parameter {name} has inf values!")


def get_data_torch(tickers, start_date, end_date, start_time, end_time, multiplier, freq, join_type='outer'):
    
    start_time = datetime.strptime(start_time, '%H:%M:%S').time()
    end_time = datetime.strptime(end_time, '%H:%M:%S').time()
    dfs = []
    
    for ticker in tqdm(tickers, desc="Getting data"):
        stock = Stock(underlying_ticker=ticker, start_date=start_date, end_date=end_date)
        stock.get_bars(multiplier, freq, adjusted=True)
        if len(stock.bars) == 0:
            continue
        else:
            df = pd.DataFrame(stock.bars)
            df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['date_time'] = df['date_time'].dt.tz_convert('America/New_York')
            df['time'] = df['date_time'].dt.time
            df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
            df = df[['timestamp', 'close']]
            df = df.rename(columns={'close': f'{ticker}_close'})
            dfs.append(df)
    
    if join_type == 'outer':
        merged_df = ft.reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), dfs)
        # merged_df.ffill(inplace=True) 
        # cannot use ffill, take DOW as an example, no price data during 2017-09-01 to 2019-04-01, ticker change on 2019-03-20 
    elif join_type == 'inner':  # default to inner join
        merged_df = ft.reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='inner'), dfs)
    else:
        raise ValueError("join_type must be either 'outer' or 'inner'")
    
    price_data_all_np = merged_df[[col for col in merged_df.columns if 'close' in col]].values
    price_data_all = torch.tensor(price_data_all_np, dtype=torch.float32)

    return price_data_all


def create_model_folder(cfg, check_exists, check_version):
    models_parent_folder = os.path.join(cfg['repo_dir'], 'models', str(cfg['multiplier']) + '_' + cfg['data_freq'])
    if check_exists and not os.path.exists(models_parent_folder):
        os.makedirs(models_parent_folder)
    
    version = 1
    model_folder_name = f"{cfg['para_name']}_v{version}"
    model_folder_path = os.path.join(models_parent_folder, model_folder_name)

    if check_version:
        while os.path.exists(model_folder_path):
            version += 1
            model_folder_name = f"{cfg['para_name']}_v{version}"
            model_folder_path = os.path.join(models_parent_folder, model_folder_name)
        else:
            os.makedirs(model_folder_path)

    return model_folder_path


def get_model_folder(cfg, version):
    models_parent_folder = os.path.join(cfg['repo_dir'], 'models', str(cfg['multiplier']) + '_' + cfg['data_freq'])
    model_folder_name = f"{cfg['para_name']}_v{version}"
    model_folder_path = os.path.join(models_parent_folder, model_folder_name)

    return model_folder_path


def save_model(model, model_folder_path, model_name):
    modele_path = os.path.join(model_folder_path, model_name)
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), modele_path)
    else:
        torch.save(model.state_dict(), modele_path)


def save_config(config, model_folder_path):

    config['optimizer_class'] = config['optimizer_class'].__name__
    config['device'] = config['device'].type

    config_file_name = 'config.json'
    config_file_path = os.path.join(model_folder_path, config_file_name)
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=4)


def save_training_plot_data_to_csv(loss_history, mean_loss, mean_reward, directory, rank):

    df = pd.DataFrame({
        'Total Loss': loss_history,
        'Mean Loss': mean_loss,
        'Mean Reward': mean_reward,
    })
    
    file_path = os.path.join(directory, 'tf_plot_data_process_' + str(rank))
    df.to_csv(file_path, index=False)
 

def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # delete 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict


def load_model_paras(model_instance, model_folder_path, model_file_name, eval_gpu='single'):
    model_path = os.path.join(model_folder_path, model_file_name)
    state_dict = torch.load(model_path)
    if eval_gpu == 'single':
        fixed_state_dict = fix_state_dict(state_dict)
    else:
        fixed_state_dict = state_dict
    model_instance.load_state_dict(fixed_state_dict)
    model_instance.eval()


def filter_pairs_with_nan(price_data, pair_indices):
    """
    Filters out asset pairs from pair_indices if any of the assets in the pair have NaN values in their price data.

    :param price_data: A tensor of shape t * n, where t is the number of timesteps and n is the number of assets.
    :param pair_indices: A list of tuples, where each tuple contains the indices of the assets to be paired.
    :return: A list of tuples representing the asset pairs that do not contain NaN values in their price data.
    """
    
    # Find the columns that contain NaN values
    columns_with_nan = torch.isnan(price_data).any(dim=0).nonzero(as_tuple=True)[0].tolist()
    filtered_pairs = [pair for pair in pair_indices if not (pair[0] in columns_with_nan or pair[1] in columns_with_nan)]
    
    return filtered_pairs


'''
def plot_loss(fig, ax, loss_history):
    # display the initial figure and get a display handle, outside the loop
    # fig, ax = plt.subplots()
    # dh = display.display(fig, display_id=True)
    ax.plot(loss_history)
    ax.set_title("Training Loss over Time")
    # ax.set_xlabel("Episode Step")
    ax.set_ylabel("Loss")


def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.title("Training Loss over Time")
    plt.xlabel("Episode Step")
    plt.ylabel("Loss")
    plt.draw()
    plt.pause(0.1) 

def plot_mean_reward(fig, ax, mean_reward_history):
    # display the initial figure and get a display handle, outside the loop
    # fig, ax = plt.subplots()
    # dh = display.display(fig, display_id=True)
    ax.plot(mean_reward_history)
    ax.set_title("Mean reward over Time")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Rewards")

def plot_batch_size(fig, ax, batch_size_history):
    # display the initial figure and get a display handle, outside the loop
    # fig, ax = plt.subplots()
    # dh = display.display(fig, display_id=True)
    ax.plot(batch_size_history)
    ax.set_title("Number of pairs retained over Time")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Pairs")
'''

def plot_and_return_top_pairs(num_quantiles, top_n_pairs_each_quantile, port_value_history, pair_names):

    # non_zero_quantity_idx = np.where(np.sum(quantity_history, axis=(0, 2)) != 0)[0]
    # non_zero_port_values = port_value_history[:, non_zero_quantity_idx]

    quantiles_values = np.linspace(1 / num_quantiles, 1, num_quantiles)
    quantiles_idx = (quantiles_values * port_value_history.shape[0]).astype(int) - 1

    top_portfolios_idx = set()
    for idx in quantiles_idx:
        top_n_port_idx = port_value_history[idx].argsort()[-top_n_pairs_each_quantile:][::-1]  # ordered
        top_portfolios_idx.update(top_n_port_idx)

    selected_pairs = set()
    plt.figure(figsize=(15, 10))
    for pair_idx in top_portfolios_idx:
        selected_pairs.add(pair_names[pair_idx])
        plt.plot(port_value_history[:, pair_idx], label=pair_names[pair_idx])
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized portfolio Value')
    plt.title('Normalized portfolio Value Over Time for Selected Asset Pairs')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    return selected_pairs


def plot_a_pair(pair_name, tickers, sampled_pairs, price_data_all_np, reg_rolling_window, port_value_history, action_history):
    ticker1, ticker2 = pair_name.split('-')
    idx1 = tickers.index(ticker1)
    idx2 = tickers.index(ticker2)
    pair_idx = sampled_pairs.index((idx1, idx2))
    price1 = price_data_all_np[:, idx1]
    price2 = price_data_all_np[:, idx2]
    spread = price1 - price2
    port_value = np.concatenate((np.ones(reg_rolling_window), port_value_history[:, pair_idx]))
    action = np.concatenate((np.zeros(reg_rolling_window), action_history[:, pair_idx]))
    action_diff = np.diff(action, prepend=action[0])

    # long_idx: index where position changes from 0 to 1 or 2 to 1
    potential_long_idx = np.where((action_diff == 1) | (action_diff == -1))[0]
    long_idx = potential_long_idx[np.where(action[potential_long_idx] == 1)]

    # short_idx: index where position changes from 0 to 2 or 1 to 2
    potential_short_idx = np.where((action_diff == 2) | (action_diff == 1))[0]
    short_idx = potential_short_idx[np.where(action[potential_short_idx] == 2)]

    # close_idx: index where position changes from 1 to 0 or 2 to 0
    potential_close_idx = np.where((action_diff == -1) | (action_diff == -2))[0]
    close_idx = potential_close_idx[np.where(action[potential_close_idx] == 0)]

    fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(price1, label=ticker1, color='#3C91E6')
    axs[0].set_title(ticker1 + ' (Y)')
    axs[0].legend()

    axs[1].plot(price2, label=ticker2, color='#4682B4')
    axs[1].set_title(ticker2 + ' (X)')
    axs[1].legend()

    axs[2].plot(spread, label="Spread: Y - X", color='black')
    axs[2].plot(long_idx, spread[long_idx], 'go', label='Long Spread')
    axs[2].plot(short_idx, spread[short_idx], 'ro', label='short Spread')
    axs[2].plot(close_idx, spread[close_idx], 'bo', label='Close Position')
    axs[2].set_title(f"Spread of {ticker1} - {ticker2}")
    axs[2].legend()

    axs[3].plot(port_value, label='Normalized Portfolio Value', color='black')
    axs[3].set_title('Normalized Portfolio Value')
    axs[3].legend()

    for ax in axs:
        ax.fill_between(range(reg_rolling_window), ax.get_ylim()[0], ax.get_ylim()[1], color='lightgrey',
                        label='Initial regression window, no trades')
    axs[-1].legend()

    plt.tight_layout()
    plt.show()


def plot_return_distribution(port_value_history, choice='total return of each pair', bins=20):
    """
    Plot the return distribution based on the choice.

    Parameters:
    - port_value_history: numpy array of shape (t, n) representing the historical prices of n assets across t time steps.
    - choice: string indicating how to calculate and plot the returns.
        - 'total return of each pair': Plot the distribution of total returns for n assets during the period.
        - 'time_steps': Consider all assets as a single equal-weighted portfolio and plot the distribution of returns across t time steps.
        - 'all': Plot the distribution of all returns for all assets across all time steps.

    """

    if choice == 'total return of each pair':
        returns = (port_value_history[-1] / port_value_history[0]) - 1
        title = 'Distribution of Total Returns of Each Pair'

    elif choice == 'across time steps':
        portfolio_values = np.sum(port_value_history, axis=1)
        returns = (portfolio_values[1:] / portfolio_values[:-1]) - 1
        title = 'Distribution of Returns for Portfolio Across Time Steps'

    elif choice == 'all':
        returns = (port_value_history[1:] / port_value_history[:-1]) - 1
        returns = returns.flatten()
        title = 'Distribution of All Returns Across All Assets and Time Steps'

    else:
        raise ValueError("Invalid choice. Choose from 'total return of each pair', 'across time steps', or 'all'.")

    plt.hist(returns, bins=bins, density=True, alpha=0.7)
    plt.title(title)
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

