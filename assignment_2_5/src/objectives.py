from elastic_energy import NeoHookeanElasticEnergy
from fem_system import FEMSystem, compute_shape_matrices
import torch

class ObjectiveBV:
    def __init__(self, vt_surf, bv):
        self.vt_surf, self.bv= vt_surf, bv
        self.length_scale = torch.linalg.norm(vt_surf.max(dim=0).values - vt_surf.min(dim=0).values)
    
    def obj(self, v):
        return objective_target_BV(v, self.vt_surf, self.bv) / (self.length_scale ** 2)
    
    def grad(self, v):
        return grad_objective_target_BV(v, self.vt_surf, self.bv) / (self.length_scale ** 2)

class ObjectiveReg:
    def __init__(self, params_init, params_idx, harm_int, weight_reg=0.0, energy_scale=1.0):
        self.params_init, self.params_idx, self.harm_int = params_init, params_idx, harm_int
        self.weight_reg, self.energy_scale = weight_reg, energy_scale
    
    def obj(self, solid, params_tmp):
        if self.weight_reg == 0: return 0
        return self.weight_reg / self.energy_scale * regularization_neo_hookean(self.params_init, solid, params_tmp, self.params_idx, self.harm_int)
    
    def grad(self, solid, params_tmp):
        if self.weight_reg == 0: return torch.zeros_like(self.params_init)
        return self.weight_reg / self.energy_scale * regularization_grad_neo_hookean(self.params_init, solid, params_tmp, self.params_idx, self.harm_int)

def objective_target_BV(v, vt, bv):
    '''
    Args:
        v: torch Tensor of shape (#v, 3), containing the current vertices position
        vt: torch Tensor of shape (#bv, 3), containing the target surface 
        bv: boundary vertices index (#bv,)
    
    Returns:
        objective: single scalar measuring the deviation from the target shape
    '''
    return 0

def grad_objective_target_BV(v, vt, bv):
    '''
    Args:
        v: torch Tensor of shape (#v, 3), containing the current vertices position
        vt: torch Tensor of shape (#bv, 3), containing the target surface 
        bv: boundary vertices (#BV,)
    
    Returns:
        gradient : torch Tensor of shape (#v, 3)
    '''
    return torch.zeros_like(v)
