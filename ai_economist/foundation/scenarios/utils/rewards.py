# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import csv

from ai_economist.foundation.scenarios.utils import social_metrics


def isoelastic_coin_minus_labor(
    coin_endowment, total_labor, isoelastic_eta, labor_coefficient
):
    """Agent utility, concave increasing in coin and linearly decreasing in labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1. 0 yields utility
            that increases linearly with coin. 1 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert 0 <= isoelastic_eta <= 1.0

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.max(1, coin_endowment))
    else:  # isoelastic_eta >= 0
        util_c = (coin_endowment ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util


def coin_minus_labor_cost(
    coin_endowment, total_labor, labor_exponent, labor_coefficient
):
    """Agent utility, linearly increasing in coin and decreasing as a power of labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        labor_exponent (float): Constant describing the shape of the utility profile
            with respect to total labor. Must be between >1.
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor.

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert labor_exponent > 1

    # Utility from coin endowment
    util_c = coin_endowment

    # Disutility from labor
    util_l = (total_labor ** labor_exponent) * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util


def coin_eq_times_productivity(coin_endowments, equality_weight):
    """Social welfare, measured as productivity scaled by the degree of coin equality.

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        equality_weight (float): Constant that determines how productivity is scaled
            by coin equality. Must be between 0 (SW = prod) and 1 (SW = prod * eq).

    Returns:
        Product of coin equality and productivity (float).
    """
    n_agents = len(coin_endowments)
    prod = social_metrics.get_productivity(coin_endowments) / n_agents
    equality = equality_weight * social_metrics.get_equality(coin_endowments) + (
        1 - equality_weight
    )
    return equality * prod


def inv_income_weighted_coin_endowments(coin_endowments):
    """Social welfare, as weighted average endowment (weighted by inverse endowment).

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.

    Returns:
        Weighted average coin endowment (float).
    """
    pareto_weights = 1 / np.maximum(coin_endowments, 1)
    pareto_weights = pareto_weights / np.sum(pareto_weights)
    return np.sum(coin_endowments * pareto_weights)


def inv_income_weighted_utility(coin_endowments, utilities):
    """Social welfare, as weighted average utility (weighted by inverse endowment).

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        utilities (ndarray): The array of utilities for each of the agents in the
            simulated economy.

    Returns:
        Weighted average utility (float).
    """
    pareto_weights = 1 / np.maximum(coin_endowments, 1)
    pareto_weights = pareto_weights / np.sum(pareto_weights)
    return np.sum(utilities * pareto_weights)


def agent_reward_total(balance, max_balance):
    reward = balance/max_balance
    # Transform total reward to be [-1,1]
    return 2*reward - 1

    
def planner_metric_stability(prices, index):
    if index > 1:
        price_diffs = [prices[i] - prices[i-1] for i in range(index)]
    elif index >= 10:
        price_diffs = [prices[i] - prices[i-1] for i in range(index-10, index)]
    else:
        return 0.0
    
    negative_diffs = [diff for diff in price_diffs if diff < 0]  # Filter only negative differences
    
    if len(negative_diffs) == 0:
        return 0.0
    
    std = np.std(negative_diffs)
    
    #file_path = 'C:\\Users\\caspe\\Desktop\\stdvs.csv'
    # Writing to CSV
    #with open(file_path, mode='a', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow([std])
    
    return std
    
def planner_metric_liquidity(volume_today, volumes, index):
    
    file_path = 'C:\\Users\\caspe\\Desktop\\volumes.csv'
    # Writing to CSV
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([volume_today])
    
    if index > 0:
        volume_average = np.average(volumes[:index])
        return volume_comparer(volume_average, volume_today)
  
    elif index >= 10:
        volume_average = np.average(volumes[index - 10:index])  
        return volume_comparer(volume_average, volume_today)

    else:
        return 0.0
    
def volume_comparer(volume_average, volume_today):
    if volume_today >= volume_average:
        return 1.0
    elif volume_average > 0.0:
        return volume_today / volume_average
    else:
        return 1.0
    
def planner_reward_total(prices, volumes, volume_today, index):
    
    std = 1 - ((planner_metric_stability(prices, index)) / (220))
    liq = planner_metric_liquidity(volume_today, volumes, index)

    reward = 0.5*std + 0.5*liq
    #print(2*reward -1)
    #print("based on liq: ", liq, " --- std: ", std)
    return 2*reward -1


def reward_function_planner(prices, index, volume, volumes, volume_weight=0.5):
    """
    Reward function to evaluate the current state of the market.
    
    Parameters:
        volume (float): Current volume of the market.
        std_dev (float): Current standard deviation of the market.
        volume_weight (float): Weight of volume in the reward calculation.
        
    Returns:
        float: Reward value.
    """
    
   # Gathered from base data
    AVERAGE_VOLUME = 27.61291121680539
    AVERAGE_STDV = 7.880280857892539
    
    # Get target values which the planner wants to achieve
    target_volume = 1.1 * AVERAGE_VOLUME
    target_std_dev = 0.9 * AVERAGE_STDV
    
    # Get standard deviation from past 5 timesteps
    std_dev = planner_metric_stability(prices, index)
    
    max_vol = max(volumes)
            
    if max_vol == 0:
        vol_reward = 0.0
    else:
        vol_reward = volume/max_vol

    # Calculate std_dev deviation from target
    std_dev_deviation = abs(std_dev - target_std_dev) / target_std_dev
    
    # Reward is a combination of volume and std_dev deviation
    reward = ((1 - volume_weight) * (1 - std_dev_deviation)) + (volume_weight * vol_reward)
    
    return reward