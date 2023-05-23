
from modules.dependencies import *


class Probabilistic_Inference:
    @staticmethod
    def run_inference(Inference_Engine, params, method='NUTS', *args, **kwargs):
        
        model = Inference_Engine.model
        method_map = {
            'NUTS': NUTS(model),
            'HMC':  HMC(model),
            'SA':   SA(model)
            }
    
        # kernel = method_map[method]
        kernel = method_map.get(method, None)
        assert kernel is not None, 'method must be in: [`NUTS`, `HMC`, `SA`]' 
        mcmc = MCMC(
            kernel,
            num_warmup   = params.num_warmup,
            num_samples  = params.num_samples,
            num_chains   = params.num_chains,
            progress_bar = False
            if ("NUMPYRO_SPHINXBUILD" in os.environ or params.disable_progbar) else True)
        try:
            mcmc.run(jax.random.PRNGKey(0), *args, **kwargs)
        except Exception as e:
            print(e)
            return args, kwargs
            # print('args:   ', args)
            # print('kwargs: ', kwargs)
        # attach to self
        try:
            Inference_Engine.mcmc = mcmc
        except Exception as e:
            print(e)
        return mcmc

    @staticmethod
    def to_jax(x):
        return jnp.asarray(x)
    
    @staticmethod
    def model_accuracy(model, samples, *args, **kwargs):
        predictive  = Predictive(model, samples)
        # probs/
        predictions = predictive(jax.random.PRNGKey(0), *args, **kwargs)['obs']
        return predictive, predictions
    
    @staticmethod
    def metrics(y, y_pred, prints=True):
        acc     = accuracy_score(y,     y_pred)
        ROC     = roc_auc_score(y,      y_pred)
        cm      = confusion_matrix(y,   y_pred)
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        uc      = auc(fpr, tpr)
        if prints:
            print('..'*10)
            print('Accuracy:    ', acc)
            print('ROC:         ', ROC)
            print('AUC:         ', uc)
            print('..'*10)
        return acc, ROC, uc, cm, fpr, tpr, thresholds