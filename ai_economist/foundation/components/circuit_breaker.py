import numpy as np
from ai_economist.foundation.base.base_component import BaseComponent, component_registry

@component_registry.add
class ExecCircuitBreaker(BaseComponent):
    name = "ExecCircuitBreaker"
    required_entities = ["AbleToBuy", "AbleToSell"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    def __init__(
        self,
        *base_component_args,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # this defines the maximum importance of the green score (20 is equal to 100%, 1 to 5%)
        self.no_actions = 2

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicPlanner":
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
        masks[self.world.planner.idx] = [1 for _ in range(self.no_actions)]
        return masks

    def component_step(self):
        planner_action = self.world.planner.get_component_action(self.name)
        print(planner_action)
        
        if 0 <= planner_action <= 1: # Make sure the planner action is legal
        
            # Let the market run its course and don't block trading
            if planner_action == 0:
                for agent in self.world.get_random_order_agents():
                    agent.state["endogenous"]["AbleToBuy"] = 0.0
                    agent.state["endogenous"]["AbleToSell"] = 0.0

            # Execute circuit breaker and block trading
            if planner_action == 1:
                for agent in self.world.get_random_order_agents():
                    agent.state["endogenous"]["AbleToBuy"] = 1.0
                    agent.state["endogenous"]["AbleToSell"] = 1.0
                
        
        else: 
            raise ValueError

    def generate_observations(self):
        obs = {}
        return obs