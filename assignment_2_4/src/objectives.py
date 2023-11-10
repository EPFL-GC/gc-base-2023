import torch

def objective_target_BV(v, vt, bv):
    '''
    Args:
        v: torch tensor of shape (#v, 3), containing the current vertices position
        vt: torch tensor of shape (#bv, 3), containing the target surface 
        bv: boundary vertices index (#bv,)
    
    Returns:
        objective: single scalar measuring the deviation from the target shape
    '''
    return 0.0