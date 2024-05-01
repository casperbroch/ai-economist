# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pandas as pd
import random
import os

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.scenarios.utils import rewards, social_metrics
from ai_economist.foundation.base.stock_market import StockMarket


@scenario_registry.add
class StockMarketSimulation(BaseEnvironment):
    name = "stock_market_simulation"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    required_entities = ["TotalBalance", "AvailableFunds", "StockPrice", "StockPriceHistory", "Demand", "Supply", "Volumes"]
    market = StockMarket("MSFT")


    def __init__(
        self,
        *base_env_args,
        **base_env_kwargs
    ):
        super().__init__(*base_env_args, **base_env_kwargs)
        self.num_agents = len(self.world.agents)
        self.curr_optimization_metrics = {str(a.idx): 0.0 for a in self.all_agents}



    # The following methods must be implemented for each scenario
    # -----------------------------------------------------------
    def reset_starting_layout(self):
        """
        Part 1/2 of scenario reset. This method handles resetting the state of the
        environment managed by the scenario (i.e. resource & landmark layout).
        We don't use resources/landmarks so this function is empty for now
        Here, generate a resource source layout consistent with target parameters.
        """

    def reset_agent_states(self):
        """
        Part 2/2 of scenario reset. This method handles resetting the state of the
        agents themselves (i.e. inventory, locations, etc.).
        Here, empty inventories, give mobile agents any starting coin, and place them
        in random accesible locations to start.
        """
        self.market = StockMarket("MSFT").simulate(1)

        self.world.clear_agent_locs()

        for agent in self.world.agents:

            # This will set all variables to 0
            agent.state["endogenous"] = {k: 0.0 for k in agent.state["endogenous"].keys()}

            starting_funds = np.random.normal(20000, 5000)
            agent.state["endogenous"]["AvailableFunds"] = starting_funds
            agent.state["endogenous"]["TotalBalance"] = starting_funds
            
            agent.state["endogenous"]["StockPrice"] = self.market.price
            agent.state["endogenous"]["StockPriceHistory"] = np.array([self.market.price])
            agent.state["endogenous"]["Volumes"] = np.array([0])

            



    def scenario_step(self):
        """
        Update the state of the world according to whatever rules this scenario
        implements.
        This gets called in the 'step' method (of base_env) after going through each
        component step and before generating observations, rewards, etc.
        """

        # Update market price
        self.market.nextstep
        
        volume = 0
        
        # Update market price within agent state
        for agent in self.world.agents:
            agent.state["endogenous"]["StockPrice"] = self.market.price
            agent.state["endogenous"]["StockPriceHistory"] = np.append(agent.state["endogenous"]["StockPriceHistory"], self.market.price)

            volume += agent.state["endogenous"]["Demand"] + agent.state["endogenous"]["Supply"]
            agent.state["endogenous"]["Volumes"] = np.append(agent.state["endogenous"]["Volumes"], volume)

        


    def generate_observations(self):
        """
        Generate observations associated with this scenario.
        A scenario does not need to produce observations and can provide observations
        for only some agent types; however, for a given agent type, it should either
        always or never yield an observation. If it does yield an observation,
        that observation should always have the same structure/sizes!
        Returns:
            obs (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent (which can including
                the planner) for which this scenario provides an observation. For each
                entry, the key specifies the index of the agent and the value contains
                its associated observation dictionary.
        Here, non-planner agents receive spatial observations (depending on the env
        config) as well as the contents of their inventory and endogenous quantities.
        The planner also receives spatial observations (again, depending on the env
        config) as well as the inventory of each of the mobile agents.
        """
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[str(agent.idx)] = {
                "Endogenous-" + k: v for k, v in agent.endogenous.items() if k != "Volumes"
            }
        
            
        prices_history = self.world.agents[0].state["endogenous"]["StockPriceHistory"]
        volumes = self.world.agents[0].state["endogenous"]["Volumes"]
        total_demand = 0.0
        total_supply = 0.0
        for agent in self.world.agents:
            total_demand += agent.state["endogenous"]["Demand"]
            total_supply += agent.state["endogenous"]["Supply"]
                

        
        obs_dict[self.world.planner.idx] = {
            "Prices_History": prices_history,
            "Total_Demand": total_demand,
            "Total_Supply": total_supply,
            "Volumes": volumes,
        }

        return obs_dict

    def compute_reward(self):
        """
        Apply the reward function(s) associated with this scenario to get the rewards
        from this step.
        Returns:
            rew (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent in the environment
                (including the planner). For each entry, the key specifies the index of
                the agent and the value contains the scalar reward earned this timestep.
        Rewards are computed as the marginal utility (agents) or marginal social
        welfare (planner) experienced on this timestep. Ignoring discounting,
        this means that agents' (planner's) objective is to maximize the utility
        (social welfare) associated with the terminal state of the episode.
        """
        curr_optimization_metrics = self.get_current_optimization_metrics(
            self.world.agents
        )
        planner_agents_rew = {
            k: v - self.curr_optimization_metrics[k]
            for k, v in curr_optimization_metrics.items()
        }
        self.curr_optimization_metrics = curr_optimization_metrics
        return planner_agents_rew

    # Optional methods for customization
    # ----------------------------------
    def additional_reset_steps(self):
        """
        Extra scenario-specific steps that should be performed at the end of the reset
        cycle.
        For each reset cycle...
            First, reset_starting_layout() and reset_agent_states() will be called.
            Second, <component>.reset() will be called for each registered component.
            Lastly, this method will be called to allow for any final customization of
            the reset cycle.
        """
        self.curr_optimization_metrics = self.get_current_optimization_metrics(
            self.world.agents
        )

    def scenario_metrics(self):
        """
        Allows the scenario to generate metrics (collected along with component metrics
        in the 'metrics' property).
        To have the scenario add metrics, this function needs to return a dictionary of
        {metric_key: value} where 'value' is a scalar (no nesting or lists!)
        Here, summarize social metrics, endowments, utilities, and labor cost annealing.
        """
        metrics = dict()

        # Log average endogenous, and utility for agents
        agent_endogenous = {}
        agent_utilities = []
        for agent in self.world.agents:
            for endogenous, quantity in agent.endogenous.items():
                if endogenous not in agent_endogenous:
                    agent_endogenous[endogenous] = []
                agent_endogenous[endogenous].append(quantity)

            agent_utilities.append(self.curr_optimization_metrics[agent.idx])

        for endogenous, quantities in agent_endogenous.items():
            metrics["endogenous/avg_agent/{}".format(endogenous)] = np.mean(quantities)

        metrics["util/avg_agent"] = np.mean(agent_utilities)

        # Log utility for the planner

        metrics["util/p"] = self.curr_optimization_metrics[self.world.planner.idx]

        return metrics

    def get_current_optimization_metrics(
        self, agents
    ):
        """
        Compute optimization metrics based on the current state. Used to compute reward.
        Returns:
            curr_optimization_metric (dict): A dictionary of {agent.idx: metric}
                with an entry for each agent (including the planner) in the env.
        """
        curr_optimization_metric = {}

        # Find the largest balance among agents and compute the today's volume
        max_balance = 0.0
        today_volume = 0.0
        for agent in agents:
            if agent.state["endogenous"]["TotalBalance"] > max_balance:
                max_balance = agent.state["endogenous"]["TotalBalance"]
                volume += agent.state["endogenous"]["Demand"] + agent.state["endogenous"]["Supply"]
        
        # Find prices and volumes arrays
        prices = self.world.agents[0].state["endogenous"]["StockPriceHistory"]
        volumes = self.world.agents[0].state["endogenous"]["Volumes"]
        

        # Optimization metric for agents:
        for agent in agents:
            curr_optimization_metric[
                agent.idx
            ] = rewards.agent_reward_total(
                agent.state["endogenous"],
                max_balance
            )
        
        # Optimization metric for the planner:
        curr_optimization_metric[
            self.world.planner.idx
        ] = rewards.planner_reward_total(
                prices,
                volumes,
                today_volume
            )
        return curr_optimization_metric