# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from ai_economist.foundation.base.registrar import Registry


class Endogenous:
    """Base class for endogenous entity classes.

    Endogenous entities are those that, conceptually, describe the internal state
    of an agent. This provides a convenient way to separate physical entities (which
    may exist in the world, be exchanged among agents, or are otherwise in principal
    observable by others) from endogenous entities (such as the amount of labor
    effort an agent has experienced).

    Endogenous entities are registered in the "endogenous" portion of an agent's
    state and should only be observable by the agent itself.
    """

    name = None

    def __init__(self):
        assert self.name is not None


endogenous_registry = Registry(Endogenous)


@endogenous_registry.add
class Labor(Endogenous):
    """Labor accumulated through working. Included in all environments by default."""

    name = "Labor"
    
@endogenous_registry.add
class TotalBalance(Endogenous):
    """The amount of money a person holds (stocks + available funds)"""

    name = "TotalBalance"
    
@endogenous_registry.add
class AvailableFunds(Endogenous):
    """The amount of available funds a person holds"""

    name = "AvailableFunds"
    
@endogenous_registry.add
class NumberOfStocks(Endogenous):
    """The amount of stocks a person holds"""

    name = "NumberOfStocks"
    
@endogenous_registry.add
class StockPrice(Endogenous):
    """The price of a stock"""

    name = "StockPrice"
    
@endogenous_registry.add
class StockPriceHistory(Endogenous):
    """The history of the price of a stock"""

    name = "StockPriceHistory"
    
@endogenous_registry.add
class Demand(Endogenous):
    """The amount of demand a user has in this timestep for a certain stock"""

    name = "Demand"

@endogenous_registry.add
class Supply(Endogenous):
    """The amount of supply a user has in this timestep for a certain stock"""

    name = "Supply"

@endogenous_registry.add
class AbleToBuy(Endogenous):
    """Determines if an agent is able to buy or not"""

    name = "AbleToBuy"
    
@endogenous_registry.add
class AbleToSell(Endogenous):
    """Determines if an agent is able to sell or not"""

    name = "AbleToSell"

@endogenous_registry.add
class Volumes(Endogenous):
    """Keeps track of the volumes (demand+supply)"""

    name = "Volumes"
    
    


