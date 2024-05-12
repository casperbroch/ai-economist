import numpy as np
import math
from ai_economist.foundation.base.base_component import BaseComponent, component_registry

@component_registry.add
class BuyOrSellStocks(BaseComponent):
    name = "BuyOrSellStocks"
    required_entities = ["TotalBalance", "AvailableFunds", "NumberOfStocks", "StockPrice", "StockPriceHistory", "Demand", "Supply", "AbleToBuy", "AbleToSell"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # TODO: determine actual price of RECs
        self.transaction_cost = 0.0075

        # this defines the maximum # of rec packages (each 5% of their energy consumption) an agent can purchase
        self.no_actions = 20

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return self.no_actions
        return None

     # def generate_masks(self, completions=0):
    #     if self.is_first_step:
    #         self.is_first_step = False
    #         if self.mask_first_step:
    #             return self.common_mask_off

    #     return self.common_mask_on

    def generate_masks(self, completions=0):
        masks = {}
        for agent in self.world.agents:
            masks[agent.idx] = np.ones(int(self.no_actions))

        return masks

    def component_step(self):
        for agent in self.world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if 0 <= action <= self.no_actions: # Agent action is legal    
                available_funds = agent.state["endogenous"]["AvailableFunds"]
                stock_price = agent.state["endogenous"]["StockPrice"]
                number_of_stocks = agent.state["endogenous"]["NumberOfStocks"]
                transaction_cost = self.transaction_cost
                able_to_buy = agent.state["endogenous"]["AbleToBuy"]
                able_to_sell = agent.state["endogenous"]["AbleToSell"]

                if action == 0: # Agent wants to keep stocks
                    agent.state["endogenous"]["Demand"] = 0.0
                    agent.state["endogenous"]["Supply"] = 0.0
                      
                elif action <= 10 and able_to_buy == 0.0: # Agent wants to buy stocks
                    # Compute what maximum amount of stocks able to buy is
                    max_stocks_buy = (available_funds - (available_funds * transaction_cost)) // stock_price
                    
                    # Compute how much stocks agent wants to buy (each integer step in action is 10%)
                    buy_percentage = action * 0.10
                    stocks_to_buy = math.floor(max_stocks_buy * buy_percentage)
                    
                    if stocks_to_buy > agent.state["endogenous"]["StocksLeft"]:
                        stocks_to_buy = agent.state["endogenous"]["StocksLeft"]
                    
                    # Compute how much it will cost and update variables
                    cost_to_buy = (stocks_to_buy * stock_price) + (stocks_to_buy * stock_price * transaction_cost)
                    agent.state["endogenous"]["AvailableFunds"] -= cost_to_buy
                    agent.state["endogenous"]["NumberOfStocks"] += stocks_to_buy
                    agent.state["endogenous"]["TotalBalance"] = agent.state["endogenous"]["AvailableFunds"] + (agent.state["endogenous"]["NumberOfStocks"] * agent.state["endogenous"]["StockPrice"] - (agent.state["endogenous"]["NumberOfStocks"] * agent.state["endogenous"]["StockPrice"] * transaction_cost))
                    agent.state["endogenous"]["Demand"] = stocks_to_buy
                    agent.state["endogenous"]["Supply"] = 0.0
                    
                    for agent in self.world.agents:
                        agent.state["endogenous"]["StocksLeft"] -= stocks_to_buy

                    
                elif 10 < action <= 20 and able_to_sell == 0.0: # Agent wants to sell stocks
                    # Compute how much stocks agent wants to sell (each integer step in action is 10%)
                    sell_percentage = (action-10) * 0.10
                    stocks_to_sell = math.floor(number_of_stocks * sell_percentage)
                    
                    # Compute profit from stock selling and cost from transaction and update variabels
                    proceeds_from_sell = stocks_to_sell * stock_price
                    transaction_cost_amount = proceeds_from_sell * transaction_cost
                    agent.state["endogenous"]["AvailableFunds"] += proceeds_from_sell - transaction_cost_amount
                    agent.state["endogenous"]["NumberOfStocks"] -= stocks_to_sell
                    agent.state["endogenous"]["TotalBalance"] = agent.state["endogenous"]["AvailableFunds"] + (agent.state["endogenous"]["NumberOfStocks"] * agent.state["endogenous"]["StockPrice"] - (agent.state["endogenous"]["NumberOfStocks"] * agent.state["endogenous"]["StockPrice"] * transaction_cost))
                    agent.state["endogenous"]["Demand"] = 0.0
                    agent.state["endogenous"]["Supply"] = stocks_to_sell
                    
                    for agent in self.world.agents:
                        agent.state["endogenous"]["StocksLeft"] += stocks_to_sell     
                
                elif able_to_buy == 1.0 and able_to_sell == 1.0:
                    agent.state["endogenous"]["TotalBalance"] -= 5000
                    agent.state["endogenous"]["Demand"] = 0.0
                    agent.state["endogenous"]["Supply"] = 0.0           

            else: # We only declared 21 actions for this agent type, so action > 21 is an error.
                raise ValueError
            
            
    def generate_observations(self):
        obs_dict = {}

        return obs_dict