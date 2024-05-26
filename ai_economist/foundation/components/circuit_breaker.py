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

        self.no_actions = 1
        self.policy_interval = 1
        
        self.default_planner_action_mask = [1 for _ in range(self.no_actions)]
        self.no_op_planner_action_mask = [0 for _ in range(self.no_actions)]

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
        if self.world.timestep % self.policy_interval == 0:
            masks[self.world.planner.idx] = self.default_planner_action_mask
        else:
            masks[self.world.planner.idx] = self.no_op_planner_action_mask
        return masks

    def component_step(self):
        planner_action = self.world.planner.get_component_action(self.name)
        print("method action ", self.world.timestep)

        if 0 <= planner_action <= self.no_actions: # Make sure the planner action is legal        
            if planner_action == 0:
                if (self.world.timestep - 1) % self.policy_interval == 0:
                    for agent in self.world.get_random_order_agents():
                        agent.state["endogenous"]["AbleToBuy"] = 0
                        agent.state["endogenous"]["AbleToSell"] = 0
                    

            if planner_action == 1:
                if (self.world.timestep - 1) % self.policy_interval == 0:
                    for agent in self.world.get_random_order_agents():
                        agent.state["endogenous"]["AbleToBuy"] = 1
                        agent.state["endogenous"]["AbleToSell"] = 1
                    
         
        else: 
            raise ValueError

    def generate_observations(self):
        obs = {}
        return obs