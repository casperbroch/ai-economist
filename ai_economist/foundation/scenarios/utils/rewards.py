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
    balance_reward = balance/max_balance
    reward = 0.5*balance_reward
    return reward 


def planner_reward_total(timestep, volumes, prices, base_volume, base_std, liq_importance=0.5):
    liq = planner_reward_liq(timestep, volumes, base_volume)
    stab = planner_reward_stab(timestep, prices, 2, base_std)
    #print("liq score ",liq)
    #print("stab score ",stab)
    
    reward = (liq_importance * liq) - ((1-liq_importance) * stab)
    return reward

def planner_reward_liq(timestep, volumes, base_volume):
    timestep -= 1
    if timestep < 1:
        curr_volume = volumes[timestep-1]
    elif timestep < 2:
        curr_volume = sum(volumes[:timestep]) / len(volumes[:timestep])
    else:
        # Otherwise, use the average of the current and the two previous volumes
        curr_volume = sum(volumes[timestep-2:timestep]) / len(volumes[timestep-2:timestep])
    
    # Determine the maximum volume so far or the base volume, whichever is greater
    max_volume = max(volumes[:timestep+1])
    
    if base_volume > max_volume:
        max_volume = base_volume
    
    if curr_volume == 0.0:
        return 0.0
    else:
        return curr_volume / max_volume
    

def planner_reward_stab(timestep, prices, window_size, base_std=5):
    if timestep < window_size - 1:
        return 0.0
    
    rolling_stds = [np.std(prices[i:i+window_size]) for i in range(timestep - window_size + 2)]
    current_std = rolling_stds[-1]
    
    max_std = max(rolling_stds)
    
    if max_std < base_std:
        max_std = base_std
    
    normalized_std = current_std / max_std if max_std != 0 else 0  # Avoid division by zero
    
    return normalized_std

