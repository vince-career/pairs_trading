import numpy as np
from itertools import combinations
import torch
from rl_functions_np import get_regression_params_np
from polygon_v9 import Stock
from tqdm.notebook import tqdm
import pandas as pd
import functools as ft
import os
import matplotlib.pyplot as plt


def cal_stats_torch(sample):
    return torch.mean(sample), torch.std(sample)


def get_regression_params_torch(y, x, device):
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=device)
    X = torch.cat([ones, x.unsqueeze(-1)], dim=1)
    weights = torch.linalg.lstsq(X, y.unsqueeze(-1))[0]
    # The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device
    intercept = weights[0][0]
    slope = weights[1][0]
    return intercept, slope


def generate_pairs_and_features_torch(price_data, k, rolling_window, device, full_sample=False, pair_indices_given=[]):
    """
    :param price_data: tensor with dim (t, n), t: timestamp, n: number of asset prices
    :param k: integer, randomly select k pairs
    :param rolling_window: integer, window to calculate normalized log price and spread
    :param device: cpu or gpu
    :param full_sample: boolean, flag to decide whether to sample all combinations or not
    :param pair_indices_given: list of tuples, indices of the price_data the agent is going to use/evaluate,
    :return: tensor with dim (t, k, 5)

    # three models:
    # 1). randomly returns k pairs from all combination
    # 2). full_sample = True, returns all possible pairs from all combination (order doesn't matter)
    # 3). agent evaluates trading performance given pair indices

    # features:
    for a specific t and k, the first 2 dim are the asset prices of a pair,
    the 3rd and 4th dim last dim are normalized log prices for rolling_window
    the last dim is the spread of the normalized log price estimated by rolling window regression.
    # spread Zt = log(price1) - slope * log(price2) - intercept
    """

    if full_sample and bool(pair_indices_given):
        raise ValueError('If you choose full_sample=True, pair_indices_given will not be used')

    t, n = price_data.shape
    log_prices = torch.log(price_data)

    if bool(pair_indices_given):
        k = len(pair_indices_given)
        pair_indices = pair_indices_given
    else:
        all_pairs = list(combinations(range(n), 2))
        if full_sample:
            k = len(all_pairs)
            pair_indices = all_pairs
        else:
            sampled = torch.randperm(len(all_pairs))[:k]
            pair_indices = [all_pairs[i] for i in sampled]

    result = torch.zeros((t, k, 5), device=device)
    for p, pair in enumerate(pair_indices):
        price1 = price_data[:, pair[0]]
        price2 = price_data[:, pair[1]]
        result[:, p, 0] = price1
        result[:, p, 1] = price2

        log_price1 = log_prices[:, pair[0]]
        log_price2 = log_prices[:, pair[1]]

        for i in range(rolling_window, t):
            sample1 = log_price1[i-rolling_window:i]
            sample2 = log_price2[i-rolling_window:i]
            mean1, std1 = cal_stats_torch(sample1)
            mean2, std2 = cal_stats_torch(sample2)

            norm_log_price1 = (log_price1[i] - mean1) / std1
            norm_log_price2 = (log_price2[i] - mean2) / std2

            # original calculation on gpu
            # intercept, slope = get_regression_params_torch(y=sample1, x=sample2, device=device)

            # section begins: send to cpu, calculate intercept and slope, then send back to gpu
            sample1_np = sample1.cpu().numpy()
            sample2_np = sample2.cpu().numpy()
            intercept_np, slope_np = get_regression_params_np(y=sample1_np, x=sample2_np)
            intercept = torch.tensor(intercept_np, dtype=torch.float32).to(device)
            slope = torch.tensor(slope_np, dtype=torch.float32).to(device)
            # section ends

            spread = log_price1[i] - (slope * log_price2[i] + intercept)    # residual, observed y - predicted y

            result[i, p, 2] = norm_log_price1
            result[i, p, 3] = norm_log_price2
            result[i, p, 4] = spread

    return result, pair_indices


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
        selected_action = torch.stack([torch.multinomial(probs, 1) for probs in action_probs]).squeeze()
    else:
        raise ValueError("Invalid method. Choose either 'greedy' or 'random'.")

    return selected_action


def discounted_rewards_torch(rewards, gamma=0.99):
    t_steps = torch.arange(rewards.size(0)).reshape(-1, 1).float().to(rewards.device)
    r = rewards * gamma**t_steps
    discounted_r = r.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]) / gamma**t_steps

    return discounted_r


def update_quant_cash_pv_no_lvg_torch(price, current_x_quantity, current_y_quantity, cash, port_value, current_position, action):
    # for optimizing the model and finding trading signal only, cannot be used in backtesting directly
    # model assumptions: no leverage: long_margin = 1, short_margin = 1, tx_cost = 0, no bid-ask spread
    # effectively, the whole portfolio will be split into two equal small portfolios for trading X and Y separately

    long_margin = 1
    short_margin = 1
    x_price = price[:, 0]
    y_price = price[:, 1]

    # close positions if necessary
    need_to_change = action != current_position
    cash_if_close = cash + current_x_quantity * x_price * long_margin + current_y_quantity * y_price * short_margin
    cash_after_close = torch.where(need_to_change, cash_if_close, cash)
    pv_after_close = torch.where(need_to_change, cash_if_close, port_value)     # if close, pv = cash

    # open new positions
    buying_power = cash_after_close * 1/2
    shorting_power = cash_after_close * 1/2

    x_quantity_long = (buying_power / x_price).floor()
    x_quantity_short = -(shorting_power / x_price).floor()
    y_quantity_long = (buying_power / y_price).floor()
    y_quantity_short = -(shorting_power / y_price).floor()

    # lxsy: long X short Y; sxly: short X long Y
    lxsy_cash = cash_after_close - x_quantity_long * x_price * long_margin - y_quantity_short * y_price * short_margin
    sxly_cash = cash_after_close - x_quantity_short * x_price * short_margin - y_quantity_long * y_price * long_margin

    new_x_quantity = torch.where(need_to_change, torch.where(action == 1, x_quantity_long, torch.where(action == 2, x_quantity_short, 0)), current_x_quantity)
    new_y_quantity = torch.where(need_to_change, torch.where(action == 1, y_quantity_short, torch.where(action == 2, y_quantity_long, 0)), current_y_quantity)
    new_cash = torch.where(need_to_change, torch.where(action == 1, lxsy_cash, torch.where(action == 2, sxly_cash, cash_after_close)), cash)

    lxsy_pv = torch.where(need_to_change, lxsy_cash, cash) + new_x_quantity * x_price * long_margin + new_y_quantity * y_price * short_margin
    sxly_pv = torch.where(need_to_change, sxly_cash, cash) + new_x_quantity * x_price * short_margin + new_y_quantity * y_price * long_margin
    new_port_value = torch.where(action == 1, lxsy_pv, torch.where(action == 2, sxly_pv, pv_after_close))

    return new_x_quantity, new_y_quantity, new_cash, new_port_value


def update_quant_cash_pv_2x_short_torch(price, current_x_quantity, current_y_quantity, cash, port_value, current_position, action):
    # all in pairs trading
    # model assumptions: long_margin = 1, short_margin = 0.5, tx_cost = 0, no bid-ask spread
    # Value Neutral: buying power = 2/3 * cash, using 2/3 * cash, shorting power = 2/3 * cash, using 1/3 * cash

    x_price = price[:, 0]
    y_price = price[:, 1]
    long_margin = 1     # if changed, buying_power and shorting_power also need to be changed
    short_margin = 0.5

    # close positions if necessary
    need_to_change = action != current_position
    cash_if_close = cash + current_x_quantity * x_price * long_margin + current_y_quantity * y_price * short_margin
    cash_after_close = torch.where(need_to_change, cash_if_close, cash)
    pv_after_close = torch.where(need_to_change, cash_if_close, port_value)     # if close, pv = cash

    # open new positions
    buying_power = cash_after_close * 1/2
    shorting_power = cash_after_close * 1/2

    x_quantity_long = (buying_power / x_price).floor()
    x_quantity_short = -(shorting_power / x_price).floor()
    y_quantity_long = (buying_power / y_price).floor()
    y_quantity_short = -(shorting_power / y_price).floor()

    # lxsy: long X short Y; sxly: short X long Y
    lxsy_cash = cash_after_close - x_quantity_long * x_price * long_margin - y_quantity_short * y_price * short_margin
    sxly_cash = cash_after_close - x_quantity_short * x_price * short_margin - y_quantity_long * y_price * long_margin

    new_x_quantity = torch.where(need_to_change, torch.where(action == 1, x_quantity_long, torch.where(action == 2, x_quantity_short, 0)), current_x_quantity)
    new_y_quantity = torch.where(need_to_change, torch.where(action == 1, y_quantity_short, torch.where(action == 2, y_quantity_long, 0)), current_y_quantity)
    new_cash = torch.where(need_to_change, torch.where(action == 1, lxsy_cash, torch.where(action == 2, sxly_cash, cash_after_close)), cash)

    lxsy_pv = torch.where(need_to_change, lxsy_cash, cash) + new_x_quantity * x_price * long_margin + new_y_quantity * y_price * short_margin
    sxly_pv = torch.where(need_to_change, sxly_cash, cash) + new_x_quantity * x_price * short_margin + new_y_quantity * y_price * long_margin
    new_port_value = torch.where(action == 1, lxsy_pv, torch.where(action == 2, sxly_pv, pv_after_close))

    return new_x_quantity, new_y_quantity, new_cash, new_port_value


def check_parameters(model):
    for name, param in model.named_parameters():
        print(name, param)
        if torch.isnan(param).any():
            print(f"Parameter {name} has NaN values!")
        if torch.isinf(param).any():
            print(f"Parameter {name} has inf values!")


def get_data_torch(tickers, start_date, end_date, multiplier, freq, device):
    dfs = []
    for ticker in tqdm(tickers, desc="getting data"):
        stock = Stock(underlying_ticker=ticker, start_date=start_date, end_date=end_date)
        stock.get_bars(multiplier, freq, adjusted=True)
        df = pd.DataFrame(stock.bars)
        df = df[['timestamp', 'close']]
        df = df.rename(columns={'close': f'{ticker}_close'})
        dfs.append(df)
    merged_df = ft.reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='inner'), dfs)
    price_data_all_np = merged_df[[col for col in merged_df.columns if 'close' in col]].values
    price_data_all = torch.tensor(price_data_all_np, dtype=torch.float32).to(device)

    return price_data_all


def save_model(model, model_folder_path, model_file_name):
    model_path = os.path.join(model_folder_path, model_file_name)
    torch.save(model.state_dict(), model_path)
    print(model_file_name, 'saved')


def load_model_paras(model_instance, model_folder_path, model_file_name):
    model_path = os.path.join(model_folder_path, model_file_name)
    model_instance.load_state_dict(torch.load(model_path))
    model_instance.eval()


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
    axs[0].set_title(ticker1)
    axs[0].legend()

    axs[1].plot(price2, label=ticker2, color='#4682B4')
    axs[1].set_title(ticker2)
    axs[1].legend()

    axs[2].plot(spread, label="Spread", color='black')
    axs[2].plot(long_idx, spread[long_idx], 'go', label='Long Spread')
    axs[2].plot(short_idx, spread[short_idx], 'ro', label='short Spread')
    axs[2].plot(close_idx, spread[close_idx], 'bo', label='Close Position')
    axs[2].set_title(f"Spread of {ticker1}-{ticker2}")
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

