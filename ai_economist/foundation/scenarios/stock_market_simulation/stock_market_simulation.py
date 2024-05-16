# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pandas as pd
import random
import os
import csv

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.scenarios.utils import rewards, social_metrics
from ai_economist.foundation.base.stock_market import StockMarket


@scenario_registry.add
class StockMarketSimulation(BaseEnvironment):
    name = "stock_market_simulation"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    required_entities = ["TotalBalance", "StockPrice"]
    market = StockMarket("MSFT")
    
    step_indicator = 0
    
    random_stock_crash_start = 0
    intensity_crash = 0
    duration_crash = 0
    crash = False
    

    def __init__(
        self,
        *base_env_args,
        liq_importance=0.5,
        stock_price_history_length=101,
        stock_quantity=200,
        static = False,
        **base_env_kwargs,
    ):
        super().__init__(*base_env_args, **base_env_kwargs)
        self.num_agents = len(self.world.agents)
        self.curr_optimization_metrics = {str(a.idx): 0.0 for a in self.all_agents}
        self.static = static
        self.liq_importance = float(liq_importance)
        self.stock_price_history_length=int(stock_price_history_length)
        self.stock_quantity=int(stock_quantity)




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
        # At the start there is no crash happening
        self.crash = False
        
        # Set time step of start of crash
        self.random_stock_crash_start = np.random.randint(90)
        
        # Set initial crash intensity (between 10% and 30%)
        self.intensity_crash = np.random.uniform(0.3, 0.9)
        
        # Set duration of crash (between 1 and 10 days)
        self.duration_crash = np.random.randint(1, 10)

        
        self.step_indicator = 0
        self.market = StockMarket("MSFT")
        self.market.simulate(1)

        self.world.clear_agent_locs()

        for agent in self.world.agents:
            # This will set all variables to 0
            agent.state["endogenous"] = {k: 0.0 for k in agent.state["endogenous"].keys()}
            
            # Set able to buy/sell to 1 at the beginning
            agent.state["endogenous"]["AbleToBuy"] = 1
            agent.state["endogenous"]["AbleToSell"] = 1
            
            # Set starting number of stocks left in rotation
            agent.state["endogenous"]["StocksLeft"] = self.stock_quantity

            # Set the starting funds of each agent
            starting_funds = np.random.normal(20000, 500)
            agent.state["endogenous"]["AvailableFunds"] = starting_funds
            agent.state["endogenous"]["TotalBalance"] = starting_funds
            agent.state["endogenous"]["StartingFunds"] = starting_funds
            
            # Update stock price, which at start is same as low and high
            agent.state["endogenous"]["StockPrice"] = self.market.getprice()
            agent.state["endogenous"]["StockPriceHigh"] = self.market.getprice()
            agent.state["endogenous"]["StockPriceLow"] = self.market.getprice()
            
            # Set first item of stockprice history to current stock price
            agent.state["endogenous"]["StockPriceHistory"] = np.zeros(self.stock_price_history_length)
            agent.state["endogenous"]["StockPriceHistory"][self.step_indicator] = self.market.getprice()

            # Set first volume to 0
            agent.state["endogenous"]["Volumes"] = np.zeros(self.stock_price_history_length)
            agent.state["endogenous"]["Volumes"][self.step_indicator] = 0

            



    def scenario_step(self):
        """
        Update the state of the world according to whatever rules this scenario
        implements.
        This gets called in the 'step' method (of base_env) after going through each
        component step and before generating observations, rewards, etc.
        """
        self.step_indicator += 1
        
        # Before anything is done, we check if a crash is starting
        if self.step_indicator == self.random_stock_crash_start:
            self.crash = True
            self.market.price = self.market.getprice() * self.intensity_crash
            
            self.duration_crash -= 1
            if self.duration_crash == 0:
                self.crash = False
        
        # Next, we check if a crash is current happening
        if self.crash == True:
            self.intensity_crash *= np.random.uniform(0.3, 0.7)
            self.market.price = self.market.getprice() * self.intensity_crash
            
            self.duration_crash -= 1
            if self.duration_crash == 0:
                self.crash = False

        
        # Get total total supply/demand in order to determine stock price
        total_supply = 0
        total_demand = 0
        for agent in self.world.agents:
            total_demand += agent.state["endogenous"]["Demand"]
            total_supply += agent.state["endogenous"]["Supply"]
        
        # Update market price
        self.market.nextstep(total_supply, total_demand, self.stock_quantity)
        #self.market.nextsteprandom()

        # Compute total volume
        volume = total_supply + total_demand
        
        # Update values for agents
        for agent in self.world.agents:
            # Update price information
            agent.state["endogenous"]["StockPrice"] = self.market.getprice()
            agent.state["endogenous"]["StockPriceHistory"][self.step_indicator] = self.market.getprice()
            agent.state["endogenous"]["StockPriceHigh"] = np.amax(agent.state["endogenous"]["StockPriceHistory"])
            agent.state["endogenous"]["StockPriceLow"] = np.amin(agent.state["endogenous"]["StockPriceHistory"][agent.state["endogenous"]["StockPriceHistory"] != 0])
            
            # Update volume information
            agent.state["endogenous"]["Volumes"][self.step_indicator] = volume
                    
        

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
        
        # OBSERVATIONS FOR AGENTS
        obs_dict = dict()
        
        total_supply = 0
        total_demand = 0
        for agent in self.world.agents:
            total_demand += agent.state["endogenous"]["Demand"]
            total_supply += agent.state["endogenous"]["Supply"]

        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                #"Endogenous-StockPriceHistory": agent.state["endogenous"]["StockPriceHistory"],
                "Endogenous-StockPrice": agent.state["endogenous"]["StockPrice"],
                "Endogenous-StockPriceHigh": agent.state["endogenous"]["StockPriceHigh"],
                "Endogenous-StockPriceLow": agent.state["endogenous"]["StockPriceLow"],
                "Endogenous-TotalBalance": agent.state["endogenous"]["TotalBalance"],
                "Endogenous-AvailableFunds": agent.state["endogenous"]["AvailableFunds"],
                "Endogenous-Demand": agent.state["endogenous"]["Demand"],
                "Endogenous-Supply": agent.state["endogenous"]["Supply"],
                "Endogenous-TotalDemand": total_demand,
                "Endogenous-TotalSupply": total_supply,
                #"Endogenous-AbleToBuy": agent.state["endogenous"]["AbleToBuy"],
                #"Endogenous-AbleToSell": agent.state["endogenous"]["AbleToSell"],
            }
        
        if not self.static:
            # OBSERVATIONS FOR PLANNER 
            avg_balance = 0.0
            for agent in self.world.agents:
                avg_balance += agent.state["endogenous"]["TotalBalance"]
            
            avg_balance = avg_balance / self.num_agents
            abl_trade = self.world.agents[0].state["endogenous"]["AbleToBuy"]
            prices = self.world.agents[0].state["endogenous"]["StockPriceHistory"]
            volumes = self.world.agents[0].state["endogenous"]["Volumes"]

            obs_dict[self.world.planner.idx] = {
                "avg_balance": avg_balance,
                "abl_trade": abl_trade,
                "prices": prices,
                "volumes": volumes,
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
            k: v
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
        
        if not self.static:
            avg_balance = 0.0
            for agent in self.world.agents:
                avg_balance += agent.state["endogenous"]["TotalBalance"]

            metrics["system/avg_balance"] = avg_balance

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

        for agent in agents:
            reward = agent.state["endogenous"]["TotalBalance"] / agent.state["endogenous"]["StartingFunds"]
            reward /= self.n_agents
            curr_optimization_metric[
                agent.idx
            ] = reward
            
        # Optimization metric for the planner:
        volumes = agents[0].state["endogenous"]["Volumes"]
        prices = agents[0].state["endogenous"]["Volumes"]

        curr_optimization_metric[
            self.world.planner.idx
        ] = rewards.planner_reward_total(self.step_indicator, volumes, prices, self.liq_importance)
        
                
        return curr_optimization_metric
    