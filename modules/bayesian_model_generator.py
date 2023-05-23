
from modules.dependencies import *
from modules.bayesian_model_interface import *

class BayesianModelEngine(Bayesian_Model_Interface):
    
    def __init__(self, runtime_params=params, *args, **kwargs):
        super().__init__(self, runtime_params, *args, **kwargs)
        self._mcmc  = None
        self._model = self.ModelGenerator
        

    @staticmethod
    def get_plate_length(y=None, Group=None, Covariates=None):
        """
        Get plate length for numpyro plate context manager.
        """
        if y is not None:
            return len(y)
        elif Group is not None:
            return len(Group)
        else:
            if Covariates is None:
                return 1
            Cov1 = list(Covariates.values())[0]
            return len(Cov1)

    @staticmethod
    def Group_Intercept(Group, Technique):
        """
        Technique:
            -> FullyPooled
            -> PartiallyPooled
            -> Unpooled
        """
        if Technique == 'FullyPooled':
            # independent of group
            rand_int = numpyro.sample('rand_int', dist.Normal(0., 1.))
            return rand_int
                
        if Technique == 'PartiallyPooled':
            mu_a    = numpyro.sample('mu_a', dist.Normal(0., 5.))
            sg_a    = numpyro.sample('sg_a', dist.HalfNormal(5.))
            n_grp = len(np.unique(Group))
            with numpyro.plate("plate_i", n_grp):
                rand_int = numpyro.sample('rand_int', dist.Normal(mu_a, sg_a))
                return rand_int[Group]

        if Technique == 'Unpooled':
            n_grp = len(np.unique(Group))
            with numpyro.plate("plate_i", n_grp):
                rand_int = numpyro.sample('rand_int', dist.Normal(0., 0.3))
                return rand_int[Group]
        
        raise ValueError('Technique must be one of [FullyPooled, PartiallyPooled, Unpooled]')

    @staticmethod
    def random_slope(Group, Xvar):
        n_grp = len(np.unique(Group))
        mu_b  = numpyro.sample('mu_b', dist.Normal(0., 5.), sample_shape=(n_grp,))
        sg_b  = numpyro.sample('sg_b', dist.HalfNormal(5.), sample_shape=(n_grp,))
        with numpyro.plate("plate_s", n_grp):
            rand_slope = numpyro.sample('rand_slope', dist.Normal(mu_b, sg_b))
            return rand_slope[Group] * Xvar


    def ModelGenerator(self, y=None, Group=None, Group_Technique='PartiallyPooled', Intercept:bool=False, RandSlopeVar=None, **Covariates):
        """
        Flexible Model Generator.
        """

        Z = 0.

        if Intercept:
            a = numpyro.sample('intercept', dist.Normal(0., 0.2))
            Z += a
        
        n_covs = len(Covariates.keys())
        if n_covs > 0:
            Beta = numpyro.sample('Beta', dist.Normal(0., 0.5), sample_shape=(n_covs,))
            for i,(k,v) in enumerate(Covariates.items()):
                Z += v.dot(Beta[i])
            
        if Group is not None:
            # random intercepts | Group
            rand_int = BayesianModelEngine.Group_Intercept(Group, Group_Technique)
            Z += rand_int
            
            # random slopes | Group 
            if RandSlopeVar is not None:    
                # RandoSlopes : RandSlopeVar=Xvar : Random Slope Indicates the Variable name to attach random slope to.
                try:
                    Z += BayesianModelEngine.random_slope(Group, Covariates[RandSlopeVar])
                except Exception as e:
                    print(e)
                    print('Covariates: ', Covariates.keys())
                

        sigma   = numpyro.sample('sigma', dist.Exponential(1.))
        n       = BayesianModelEngine.get_plate_length(y=y, Group=Group, Covariates=Covariates)
        
        with numpyro.plate('data', n):
            numpyro.sample('obs', dist.Normal(Z, sigma), obs=y)

    

    def modelX(self, y=None, X1=None, Group=None):
        μ_α = numpyro.sample("μ_α", dist.Normal(0.0, 500.0))
        σ_α = numpyro.sample("σ_α", dist.HalfNormal(100.0))
        μ_β = numpyro.sample("μ_β", dist.Normal(0.0, 3.0))
        σ_β = numpyro.sample("σ_β", dist.HalfNormal(3.0))

        n_patients = len(np.unique(X1))

        with numpyro.plate("plate_i", n_patients):
            α = numpyro.sample("α", dist.Normal(μ_α, σ_α))
            β = numpyro.sample("β", dist.Normal(μ_β, σ_β))

        σ = numpyro.sample("σ", dist.HalfNormal(100.0))
        FVC_est = α[Group] + β[Group] * X1

        with numpyro.plate("data", len(X1)):
            numpyro.sample("obs", dist.Normal(FVC_est, σ), obs=y)