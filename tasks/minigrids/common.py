from enum import IntEnum
from typing import List

from minigrid.core.grid import Grid



class Actions(IntEnum):
    
    left = 0
    right = 1
    forward = 2
    
    
    
    
    
    
    

    
    



def _get_valid_pos(grid: Grid, agent_pos=None) -> List[tuple]:
    route_idx = [idx for idx, g in enumerate(grid.grid) if g is None]
    all_valid_pos = [(ridx % grid.width, ridx // grid.width)
                     for ridx in route_idx]
    assert all([grid.get(*ap) is None for ap in all_valid_pos]) is True
    if agent_pos is not None:
        agent_pos = tuple(agent_pos)
        if agent_pos != (-1, -1):
            all_valid_pos.remove(tuple(agent_pos))
    return all_valid_pos
