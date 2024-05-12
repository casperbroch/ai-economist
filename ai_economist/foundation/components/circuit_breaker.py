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

        self.no_actions = 2

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicPlanner":
            return self.no_actions
        return None

    def generate_masks(self, completions=0):
        masks = {}
        masks[self.world.planner.idx] = np.ones(int(self.no_actions))
        return masks

    def component_step(self):
        planner_action = self.world.planner.get_component_action(self.name)
 
        
        if 0 <= planner_action <= self.no_actions: # Make sure the planner action is legal        
            if planner_action == 0:
                # Agent does nothing
                pass
            # Let the market run its course and don't block trading
            if planner_action == 1:
                for agent in self.world.get_random_order_agents():
                    agent.state["endogenous"]["AbleToBuy"] = 0
                    agent.state["endogenous"]["AbleToSell"] = 0

            # Execute circuit breaker and block trading
            if planner_action == 2:
                for agent in self.world.get_random_order_agents():
                    agent.state["endogenous"]["AbleToBuy"] = 1
                    agent.state["endogenous"]["AbleToSell"] = 1
         
        else: 
            raise ValueError

    def generate_observations(self):
        obs = {}
        return obs