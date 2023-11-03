from fem_system import compute_barycenters
import matplotlib.pyplot as plt
import time
import torch

def compute_inverse_approximate_hessian_matrix(sk, yk, invB_prev):
    '''
    Args:
        sk: previous step x_{k+1} - x_k, shape (n, 1)
        yk: grad(f)_{k+1} - grad(f)_{k}, shape (n, 1)
        invB_prev: previous Hessian estimate Bk, shape (n, n)
    
    Returns:
        invB_new : previous Hessian estimate Bk, shape (n, n)
    '''
    invB_new  = invB_prev.clone()
    invB_new += (sk.T @ yk + yk.T @ invB_prev @ yk) / ((sk.T @ yk) ** 2) * (sk @ sk.T)
    prod      = (invB_prev @ yk) @ sk.T
    invB_new -= (prod + prod.T) / (sk.T @ yk)
    return invB_new


def equilibrium_convergence_report_GD(solid, v_init, n_steps, step_size, thresh=1.0):
    '''
    Finds the equilibrium by minimizing the total energy using gradient descent.

    Args:
        solid: an elastic solid to optimize
        v_init: the initial guess for the equilibrium position
        n_step: number of optimization steps
        step_size: scaling factor of the gradient when taking the step
        thresh: threshold to stop the optimization process on the gradient's magnitude

    Returns:
        report : a dictionary containing various quantities of interest
    '''

    energies_el  = torch.zeros(size=(n_steps+1,))
    energies_ext = torch.zeros(size=(n_steps+1,))
    residuals    = torch.zeros(size=(n_steps+1,))
    times        = torch.zeros(size=(n_steps+1,))
    step_sizes   = torch.zeros(size=(n_steps,))

    v_def_prev = v_init.clone()
    jac_tmp = solid.compute_jacobians(v_def_prev)
    strain_tmp = solid.compute_strain_tensor(jac_tmp)
    def_barycenters = compute_barycenters(v_def_prev, solid.tet)
    f_vol_tmp, f_ext_tmp = solid.compute_volumetric_and_external_forces()
    f_el_tmp = solid.compute_elastic_forces(jac_tmp, strain_tmp)

    energies_el[0]  = solid.compute_elastic_energy(jac_tmp, strain_tmp)
    energies_ext[0] = solid.compute_external_energy(def_barycenters, f_vol_tmp)
    residuals[0]    = torch.linalg.norm((f_el_tmp + f_ext_tmp)[solid.free_idx, :])
    idx_stop        = n_steps

    energy_tot_prev = energies_el[0] + energies_ext[0]

    t_start = time.time()
    for i in range(n_steps):
        ## TODO: Find the descent direction
        ## descent_dir has shape (#v, 3)
        descent_dir = ...

        step_size_tmp  = step_size
        max_l_iter = 20
        for l_iter in range(max_l_iter):
            step_size_tmp *= 0.5
            v_def_tmp = solid.compute_pinned_deformation(v_def_prev + step_size_tmp * descent_dir)
            jac_tmp = solid.compute_jacobians(v_def_tmp)
            strain_tmp = solid.compute_strain_tensor(jac_tmp)
            def_barycenters_tmp = compute_barycenters(v_def_tmp, solid.tet)

            ## TODO: Check if the armijo rule is satisfied
            ## energy_tot_tmp is the current total energy
            ## grad_tmp is the current total energy gradient
            ## armijo is a boolean that says whether the condition is satisfied
            energy_tot_tmp = ...
            grad_tmp       = ...
            armijo         = ...
            
            if armijo or l_iter == max_l_iter-1:
                v_def_prev = v_def_tmp.clone()
                break

        step_sizes[i] = step_size_tmp

        # Measure the force residuals
        energies_el[i+1]  = solid.compute_elastic_energy(jac_tmp, strain_tmp)
        energies_ext[i+1] = solid.compute_external_energy(def_barycenters_tmp, f_vol_tmp)
        residuals[i+1]    = torch.linalg.norm((f_el_tmp + f_ext_tmp)[solid.free_idx, :])
        energy_tot_prev   = energy_tot_tmp
        
        if residuals[i+1] < thresh:
            energies_el[i+1:]  = energies_el[i+1]
            energies_ext[i+1:] = energies_ext[i+1]
            residuals[i+1:]    = residuals[i+1]
            idx_stop = i
            break
            
        times[i+1] = time.time() - t_start
            
    report = {}
    report['final_def']    = v_def_prev
    report['energies_el']  = energies_el
    report['energies_ext'] = energies_ext
    report['residuals']    = residuals
    report['times']        = times
    report['idx_stop']     = idx_stop
    report['step_sizes']   = step_sizes

    return report


def equilibrium_convergence_report_BFGS(solid, v_init, n_steps, step_size, thresh=1.0):
    '''
    Finds the equilibrium by minimizing the total energy using BFGS.

    Args:
        solid: an elastic solid to optimize
        v_init: the initial guess for the equilibrium position
        n_step: number of optimization steps
        step_size: scaling factor of the direction when taking the step
        thresh: threshold to stop the optimization process on the gradient's magnitude

    Ouput:
        report: a dictionary containing various quantities of interest
    '''

    energies_el  = torch.zeros(size=(n_steps+1,))
    energies_ext = torch.zeros(size=(n_steps+1,))
    residuals    = torch.zeros(size=(n_steps+1,))
    times        = torch.zeros(size=(n_steps+1,))
    
    v_def = v_init.clone()
    jac_tmp = solid.compute_jacobians(v_def)
    strain_tmp = solid.compute_strain_tensor(jac_tmp)
    def_barycenters = compute_barycenters(v_def, solid.tet)
    f_vol_tmp, f_ext_tmp = solid.compute_volumetric_and_external_forces()
    f_el_tmp = solid.compute_elastic_forces(jac_tmp, strain_tmp)

    energies_el[0]  = solid.compute_elastic_energy(jac_tmp, strain_tmp)
    energies_ext[0] = solid.compute_external_energy(def_barycenters, f_vol_tmp)
    ## TODO: Collect free vertex positions
    ## grad_tmp is the current flattened gradient of the total energy
    ##          with respect to the free vertices
    grad_tmp        = ...
    residuals[0]    = torch.linalg.norm(grad_tmp)
    idx_stop        = n_steps

    energy_tot_prev = energies_el[0] + energies_ext[0]

    ## TODO: Collect free vertex positions
    ## v_tmp_full are the current unflattened vertices
    ## v_tmp are the current flattened free vertices
    v_tmp_full = ...
    v_tmp = ...
    dir_zeros = torch.zeros_like(solid.v_rest)
    invB_prev = torch.eye(v_tmp.shape[0])

    t_start = time.time()
    for i in range(n_steps):

        dir_tmp = - invB_prev @ grad_tmp
        dir_zeros[solid.free_idx, :] = dir_tmp.reshape(-1, 3)

        step_size_tmp  = step_size
        max_l_iter = 20
        for l_iter in range(max_l_iter):
            step_size_tmp   *= 0.5
            v_tmp_pinned = solid.compute_pinned_deformation(v_tmp_full + step_size_tmp * dir_zeros)
            jac_tmp = solid.compute_jacobians(v_tmp_pinned)
            strain_tmp = solid.compute_strain_tensor(jac_tmp)
            def_barycenters_tmp = compute_barycenters(v_tmp_pinned, solid.tet)

            ## TODO: Check if the armijo rule is satisfied
            ## energy_tot_tmp is the current total energy
            ## armijo is a boolean that says whether the condition is satisfied
            energy_tot_tmp = ...
            armijo         = ...
            
            if armijo or l_iter == max_l_iter - 1:
                break
        
        ## TODO: Update all quantities
        ## v_new are the new flattened free vertices
        ## grad_new is the new flattened gradient of the total energy
        ##          with respect to the free vertices
        v_new     = ...
        f_vol_new, f_ext_new = solid.compute_volumetric_and_external_forces()
        grad_new  = ...
        invB_prev = compute_inverse_approximate_hessian_matrix(v_new - v_tmp, 
                                                               grad_new - grad_tmp,
                                                               invB_prev)
        v_tmp      = v_new.clone()
        v_tmp_full = v_tmp_pinned.clone()
        grad_tmp   = grad_new.clone()

        energies_el[i+1]  = solid.compute_elastic_energy(jac_tmp, strain_tmp)
        energies_ext[i+1] = solid.compute_external_energy(def_barycenters_tmp, f_vol_new)
        residuals[i+1]    = torch.linalg.norm(grad_tmp)
        
        if residuals[i+1] < thresh:
            residuals[i+1:]    = residuals[i+1]
            energies_el[i+1:]  = energies_el[i+1]
            energies_ext[i+1:] = energies_ext[i+1]
            idx_stop = i
            break
            
        times[i+1] = time.time() - t_start
            
    report = {}
    report['final_def']    = v_tmp_full
    report['energies_el']  = energies_el
    report['energies_ext'] = energies_ext
    report['residuals']    = residuals
    report['times']        = times
    report['idx_stop']     = idx_stop
    report['invB_final']   = invB_prev

    return report

def fd_validation_elastic(solid, v_def):
    torch.manual_seed(0)
    epsilons = torch.logspace(-9.0, -3.0, 100)
    perturb_global = 2.0e-3 * torch.rand(size=solid.v_rest.shape) - 1.0e-3
    v_def_perturb = solid.compute_pinned_deformation(v_def + perturb_global)
    perturb = 2.0 * torch.rand(size=solid.v_rest.shape) - 1.0
    perturb[solid.pin_idx] = 0.0
    
    # Back to original
    jac = solid.compute_jacobians(v_def_perturb)
    strain = solid.compute_strain_tensor(jac)
    f_el = solid.compute_elastic_forces(jac, strain)
    grad = -f_el
    grad[solid.pin_idx] = 0.0
    an_delta_E = (grad * perturb).sum()

    errors = []
    for eps in epsilons:
        # One step forward
        v_def_1 = v_def_perturb + eps * perturb
        jac_1 = solid.compute_jacobians(v_def_1)
        strain_1 = solid.compute_strain_tensor(jac_1)
        E1 = solid.compute_elastic_energy(jac_1, strain_1)

        # Two steps backward
        v_def_2 = v_def_perturb - eps * perturb
        jac_2 = solid.compute_jacobians(v_def_2)
        strain_2 = solid.compute_strain_tensor(jac_2)
        E2 = solid.compute_elastic_energy(jac_2, strain_2)

        fd_delta_E = (E1 - E2) / (2.0 * eps)    
        errors.append(abs(fd_delta_E - an_delta_E) / abs(an_delta_E))

    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()

def fd_validation_ext(solid, v_def):
    torch.manual_seed(0)
    epsilons = torch.logspace(-9.0, -3.0, 100)
    perturb_global = 2.0e-3 * torch.rand(size=solid.v_rest.shape) - 1.0e-3
    v_def_perturb = solid.compute_pinned_deformation(v_def + perturb_global)
    perturb = 2.0 * torch.rand(size=solid.v_rest.shape) - 1.0
    perturb[solid.pin_idx] = 0.0
    
    # Back to original
    f_vol, f_ext = solid.compute_volumetric_and_external_forces()
    grad = -f_ext
    grad[solid.pin_idx] = 0.0
    an_delta_E = (grad * perturb).sum()
    
    errors = []
    for eps in epsilons:
        # One step forward
        v_def_1 = v_def_perturb + eps * perturb
        def_barycenters_1 = compute_barycenters(v_def_1, solid.tet)
        E1 = solid.compute_external_energy(def_barycenters_1, f_vol)

        # One step backward
        v_def_2 = v_def_perturb - eps * perturb
        def_barycenters_2 = compute_barycenters(v_def_2, solid.tet)
        E2 = solid.compute_external_energy(def_barycenters_2, f_vol)

        fd_delta_E = (E1 - E2) / (2.0 * eps)    
        errors.append(abs(fd_delta_E - an_delta_E) / abs(an_delta_E))
    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()
