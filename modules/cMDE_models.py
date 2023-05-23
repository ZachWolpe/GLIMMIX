
from modules.bayesian_model_interface import *
from modules.dependencies import *

class cMDE_model(Bayesian_Model_Interface):

    def __init__(self, model='RandomIntercept', runtime_params=params, *args, **kwargs):
        model_keys = {
            'RandomIntercept':          self.RandomIntercept,
            'FullModel':                self.FullModel,
            'DynamicIntercept':         self.DynamicRandomInterceptModel,
            'DynamicSlope':             self.DynamicSlopeModel,
            'DynamicInterceptSlope':    self.DynamicInterceptSlopeModel
        }

        model = model_keys.get(model, None)
        if model is None:
            raise ValueError('Model not found. Use one of: [`RandomIntercept`, `FullModel`]')
        super().__init__(model, runtime_params, *args, **kwargs)

        
    def RandomIntercept(self, cMDE:jnp.ndarray=None, Age:jnp.ndarray=None, Group:jnp.ndarray=None):
        """
        Model: Log odds of depression (cMDE) as a linear combination of covariate : Age.
        Specify a Normal prior over regression coefficients.

        Parameters
        ----------
            cMDE (jnp.ndarray): Binary response variable. Depression determination.
            Age  (jnp.ndarray): Continuous covariate. Age of participant.
        """
        # beta = numpyro.sample('coefficients', dist.MultivariateNormal(loc=0., covariance_matrix=jnp.eye(Age.shape[1])))
        betas  = numpyro.sample('betas', dist.Normal(jnp.zeros((2)), 1.))
        n_grp  = len(np.unique(Group))
        grp_v  = numpyro.sample('grp_v', dist.Normal(jnp.zeros((n_grp)), 1.)) 
        # logits = betas[0] + betas[1]*Age + grp_v[Group] # *Age.dot(betas[1])
        logits = betas[0] + Age.dot(betas[1]) + grp_v[Group] # 

        probs  = expit(logits)
        numpyro.sample('obs', dist.Bernoulli(probs=probs), obs=cMDE)
        # with numpyro.plate('data', Age.shape[0]):
        #     numpyro.sample('obs', dist.Bernoulli(probs=probs), obs=cMDE) # logits=logits



    def FullModel(self, *args, **kwargs):
        """
        Model: 
        cMDE ~ Age + BMI + Education + Heart_Disease_Present + CurrentMedications_betablockersantiarrhythmic + (1 | Subject_ID)
        """
        betas  = numpyro.sample('betas', dist.Normal(jnp.zeros((6)), 1.))
        n_grp  = len(np.unique(kwargs['Group']))
        grp_v  = numpyro.sample('grp_v', dist.Normal(jnp.zeros((n_grp)), 1.)) # Intercept per group
        logits = betas[0]                           + \
                kwargs['Age'].dot(betas[1])         + \
                kwargs['BMI'].dot(betas[2])         + \
                kwargs['Education'].dot(betas[3])   + \
                kwargs['Heart_Disease_Present'].dot(betas[4]) + \
                grp_v[kwargs['Group']]

                # kwargs['CurrentMedications_betablockersantiarrhythmic'].dot(betas[5]) + \

        probs  = expit(logits)
        numpyro.sample('obs', dist.Bernoulli(probs=probs), obs=kwargs['cMDE'])
        
    def DynamicRandomInterceptModel(self, Group, y=None, **Covariates):
        """
        Model:
            y ~ Beta * Kwargs[Vars] + (1 | Group)
        """
        n_covars = len(Covariates.keys()) + 1
        betas    = numpyro.sample('betas', dist.Normal(jnp.zeros((n_covars)), 1.))

        n_grp  = len(np.unique(Group))
        grp_v  = numpyro.sample('grp_v', dist.Normal(jnp.zeros((n_grp)), 1.))


        logits   = betas[0]
        for i,(k,v) in enumerate(Covariates.items()):
            logits += v.dot(betas[i]) 
        logits += grp_v[Group]

        probs  = expit(logits)
        numpyro.sample('obs', dist.Bernoulli(probs=probs), obs=y)
    
    def DynamicSlopeModel(self, Group, y=None, **Covariates):
        pass
        # v.dot(betas[i])

        """
        Model:
            y ~ Beta * Kwargs[Vars] + (1 | Group)
        """
        n_covars = len(Covariates.keys()) + 1
        betas    = numpyro.sample('betas', dist.Normal(jnp.zeros((n_covars)), 1.))

        n_grp  = len(np.unique(Group))
        grp_v  = numpyro.sample('grp_v', dist.Normal(jnp.zeros((n_grp)), 1.))


        logits   = betas[0]
        for i,(_,v) in enumerate(Covariates.items()):
            logits += v.dot(betas[i])  
        logits += grp_v[Group]

        # probs  = expit(logits)
        # numpyro.sample('obs', dist.Bernoulli(probs=probs), obs=y)

    def DynamicInterceptSlopeModel(self):
        pass

    def DynamicPooledModel(self, y, Group, **Covariates):
        mu_a    = numpyro.sample('mu_a', dist.Normal(0., 100.))
        sg_a    = numpyro.sample('sg_a', dist.HalfNormal(100.))
        n_grp   = len(np.unique(Group))

        with numpyro.plate("plate_i", n_grp):
            beta = numpyro.sample("beta", dist.Normal(mu_a, sg_a))

        sigma = numpyro.sample('sigma', dist.HalfNormal(100.))
        alpha = numpyro.sample('alpha', dist.Normal(0., 1.))
        logits = alpha
        for i,(_,v) in enumerate(Covariates.items()):
            logits += v.dot(beta[i])
        
        probs = expit(logits)
        numpyro.sample('obs', dist.Bernoulli(probs=probs), obs=y)




    # def model(patient_code, Weeks, FVC_obs=None):
    #     μ_α = numpyro.sample("μ_α", dist.Normal(0.0, 500.0))
    #     σ_α = numpyro.sample("σ_α", dist.HalfNormal(100.0))
    #     μ_β = numpyro.sample("μ_β", dist.Normal(0.0, 3.0))
    #     σ_β = numpyro.sample("σ_β", dist.HalfNormal(3.0))

    #     n_patients = len(np.unique(patient_code))

    #     with numpyro.plate("plate_i", n_patients):
    #         α = numpyro.sample("α", dist.Normal(μ_α, σ_α))
    #         β = numpyro.sample("β", dist.Normal(μ_β, σ_β))

    #     σ = numpyro.sample("σ", dist.HalfNormal(100.0))
    #     FVC_est = α[patient_code] + β[patient_code] * Weeks

    #     with numpyro.plate("data", len(patient_code)):
    #         numpyro.sample("obs", dist.Normal(FVC_est, σ), obs=FVC_obs)