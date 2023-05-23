from modules.dependencies import *

class params:
    num_warmup      = 500
    num_samples     = 1000
    num_chains      = 2
    disable_progbar = False
    
    
class Bayesian_Model_Interface(ABC):
    def __init__(self, model, runtime_params, *args, **kwargs):
        self.runtime_params = runtime_params
        self._model         = model
        self._mcmc          = None

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model

    @property
    def mcmc(self):
        if self._mcmc is None:
            raise ValueError('Model not fit. Run .fit() first.')
        return self._mcmc
    
    @mcmc.setter
    def mcmc(self, value):
        if not isinstance(value, MCMC):
            raise ValueError('Value must be numpyro MCMC object.')
        self._mcmc = value

    
    def compute_model_accuracy():
        pass

    def print_summary(self):
        self.mcmc.print_summary()
    
    def plot_trace(self):
        az.plot_trace(self.mcmc)

