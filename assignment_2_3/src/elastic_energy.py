import torch

class ElasticEnergy:
    def __init__(self, young, poisson):
        '''
        Args:
            young: Young's modulus [Pa]
            poisson: Poisson ratio
        '''
        self.young = young
        self.poisson = poisson
        self.lbda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.mu = young / (2.0 * (1.0 + poisson))
        
    def compute_strain_tensor(self, jac):
        '''
        This method computes the strain tensor (#t, 3, 3), and stores it in self.E

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
        
        Returns:
            E: strain induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def compute_energy_density(self, jac):
        '''
        This method computes the energy density at each tetrahedron (#t,),
        and stores the result in self.psi

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
        
        Returns:
            psi: energy density per tet (#t,)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def compute_piola_kirchhoff_stress_tensor(self, jac):
        '''
        This method computes the stress tensor (#t, 3, 3), and stores it in self.P

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
        
        Returns:
            P: stress tensor induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

class LinearElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
    
    def compute_strain_tensor(self, jac):
        pass

    def compute_energy_density(self, jac, E):
        pass
    
    def compute_piola_kirchhoff_stress_tensor(self, jac, E):
        pass

class NeoHookeanElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
        self.muNH = self.mu
        self.lbdaNH = self.lbda + self.muNH
        self.nuNH = (self.lbdaNH - self.muNH) / (2.0 * self.lbdaNH)
        self.alpha = 0.0 # TODO: compute alpha so that the model is rest-stable
        self.psi0 = 0.0 # TODO: compute psi0 so that the model has 0 energy at rest

    def compute_strain_tensor(self, jac):
        pass

    def compute_energy_density(self, jac, E):
        pass

    def compute_piola_kirchhoff_stress_tensor(self, jac, E):
        pass
