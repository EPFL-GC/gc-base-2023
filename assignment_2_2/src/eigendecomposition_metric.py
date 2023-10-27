from fem_system import compute_barycenters
import igl
import meshplot as mp
import torch

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

def compute_eigendecomposition_metric(jac):
    '''
    Args:
        jac: a torch tensor of shape (#t, 3, 3) containing the stacked jacobians

    Returns:
        square_root_eigvals: torch tensor of shape (#t, 3) containing the square root of the eigenvalues of the metric tensor in ascending order
        eigvecs: torch tensor of shape (#t, 3, 3) containing the eigenvectors of the metric tensor
    '''

    eigvals, eigvecs = torch.zeros(size=jac.shape[:2]), torch.zeros_like(jac)

    return eigvals, eigvecs

def plot_eigendecomposition_metric(solid, v_def, square_root_eigvals, eigvecs, rot, scale=0.05):
    '''
    Args:
        solid: an FEMSystem object
        v_def: a torch tensor of shape (#v, 3) containing the deformed positions of the vertices
        square_root_eigvals : torch tensor of shape (#t, 3) containing the square root of the eigenvalues of the metric tensor
        eigvecs: torch tensor of shape (#t, 3, 3) containing the eigenvectors of the metric tensor
        rot: a rotation matrix for plotting purposes
        scale: scaling for plotting purposes
    '''

    scaled_eigvecs = scale * torch.einsum('ik, ijk -> ijk', square_root_eigvals, eigvecs)
    
    def_barycenters = compute_barycenters(v_def, solid.tet)

    start_plot0 = (def_barycenters - scaled_eigvecs[..., 0]) @ rot.T
    start_plot1 = (def_barycenters - scaled_eigvecs[..., 1]) @ rot.T
    start_plot2 = (def_barycenters - scaled_eigvecs[..., 2]) @ rot.T
    end_plot0   = (def_barycenters + scaled_eigvecs[..., 0]) @ rot.T
    end_plot1   = (def_barycenters + scaled_eigvecs[..., 1]) @ rot.T
    end_plot2   = (def_barycenters + scaled_eigvecs[..., 2]) @ rot.T

    # Get boundary edges
    be = igl.edges(igl.boundary_facets(to_numpy(solid.tet)))

    p = mp.plot(to_numpy(v_def @ rot.T), be, shading={"line_color": "black"})
    p.add_points(to_numpy(def_barycenters @ rot.T), shading={"point_color":"black", "point_size": 0.2})
    
    # In tension
    tens0 = torch.argwhere(square_root_eigvals[:, 0]>1.0 + 1.0e-6)
    tens1 = torch.argwhere(square_root_eigvals[:, 1]>1.0 + 1.0e-6)
    tens2 = torch.argwhere(square_root_eigvals[:, 2]>1.0 + 1.0e-6)
    if tens0.shape[0] != 0:
        p.add_lines(start_plot0[tens0, :], 
                    end_plot0[tens0, :], 
                    shading={"line_color": "#182C94"})
    if tens1.shape[0] != 0:
        p.add_lines(start_plot1[tens1, :], 
                    end_plot1[tens1, :], 
                    shading={"line_color": "#182C94"})
    if tens2.shape[0] != 0:
        p.add_lines(start_plot2[tens2, :], 
                    end_plot2[tens2, :], 
                    shading={"line_color": "#182C94"})

    # In compression
    comp0 = torch.argwhere(square_root_eigvals[:, 0]<1.0 - 1.0e-6)
    comp1 = torch.argwhere(square_root_eigvals[:, 1]<1.0 - 1.0e-6)
    comp2 = torch.argwhere(square_root_eigvals[:, 2]<1.0 - 1.0e-6)
    if comp0.shape[0] != 0:
        p.add_lines(start_plot0[comp0, :], 
                    end_plot0[comp0, :], 
                    shading={"line_color": "#892623"})
    if comp1.shape[0] != 0:
        p.add_lines(start_plot1[comp1, :], 
                    end_plot1[comp1, :], 
                    shading={"line_color": "#892623"})
    if comp2.shape[0] != 0:
        p.add_lines(start_plot2[comp2, :], 
                    end_plot2[comp2, :], 
                    shading={"line_color": "#892623"})

    # Neutral
    neut0 = torch.argwhere(abs(square_root_eigvals[:, 0]-1.0) < 1.0e-6)
    neut1 = torch.argwhere(abs(square_root_eigvals[:, 1]-1.0) < 1.0e-6)
    neut2 = torch.argwhere(abs(square_root_eigvals[:, 2]-1.0) < 1.0e-6)
    if neut0.shape[0] != 0:
        p.add_lines(start_plot0[neut0, :], 
                    end_plot0[neut0, :], 
                    shading={"line_color": "#027337"})
    if neut1.shape[0] != 0:
        p.add_lines(start_plot1[neut1, :], 
                    end_plot1[neut1, :], 
                    shading={"line_color": "#027337"})
    if neut2.shape[0] != 0:
        p.add_lines(start_plot2[neut2, :], 
                    end_plot2[neut2, :], 
                    shading={"line_color": "#027337"})