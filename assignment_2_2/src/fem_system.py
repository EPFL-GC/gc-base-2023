import torch

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             FEM SYSTEM CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class FEMSystem():
    
    def __init__(self, v_rest, tet, rho=1.0, pin_idx=None):
        '''
        Args:
            v_rest: rest position of the vertices of the mesh (#v, 3)
            tet: indices of the element's vertices (#t, 4)
            rho: mass per unit volume [kg.m-3]
            pin_idx: list of vertex indices to pin
        '''

        self.v_rest = v_rest.clone()
        self.tet = tet.clone()
        self.rho = rho
        if pin_idx is None:
            pin_idx = []
        self.pin_idx = torch.tensor(pin_idx)
        self.free_idx = None
        self.pin_mask = None
        self.make_free_indices_and_free_mask()

        self.rest_barycenters = None
        self.Dm = None
        self.Bm = None
        self.W0 = None
        
        self.update_rest_shape(self.v_rest)
        
    def make_free_indices_and_free_mask(self):
        '''
        Updated attributes:
            free_index: torch tensor of shape (#free_vertices,) containing the list of unpinned vertices
            free_mask: torch tensor of shape (#v, 1) containing 1 at free vertex indices and 0 at pinned vertex indices
        '''
        pass
        
    def update_rest_shape(self, v_rest):
        '''
        Args:
            v_rest: rest position of the vertices of the mesh (#v, 3)
        
        Updated attributes:
            v_rest: rest position of the vertices of the mesh (#v, 3)
            rest_barycenters: barycenters of the tetrahedra (#t, 3)
            Dm: shape matrices of the undeformed configuration (#t, 3, 3)
            Bm: inverse of the shape matrices of the undeformed configuration (#t, 3, 3)
            W0: signed volumes of the tetrahedra (#t,)
        '''
        pass
        
    def compute_pinned_deformation(self, v_def):
        '''
        Args:
            v_def: deformed position of the vertices of the mesh (#v, 3)

        Returns:
            v_def_pinned: deformed position of the vertices of the mesh after taking pinning into account (#v, 3)
        '''
        pass
        
    def compute_jacobians(self, v_def):
        '''
        Args:
            v_def: deformed position of the vertices of the mesh (#v, 3)

        Returns:
            jac: Jacobians of the deformation (#t, 3, 3)
        '''
        pass

def compute_barycenters(v, tet):
    '''
    Args:
        v: vertices of the mesh (#v, 3)
        tet: indices of the element's vertices (#t, 4)
        
    Returns:
        barycenters: barycenters of the tetrahedra
    '''
    pass

def compute_shape_matrices(v, tet):
    '''
    Args:
        v: vertices of the mesh (#v, 3)
        tet: indices of the element's vertices (#t, 4)

    Returns:
        D: shape matrices of current configuration (#t, 3, 3)
    '''
    pass

def compute_signed_volume(D):
    '''
    Args:
        D: shape matrices of current configuration (#t, 3, 3)

    Returns:
        signed_volume: signed volumes of the tetrahedra (#t,)
    '''
    pass
