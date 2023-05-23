
from modules.dependencies import *


class FilterData:
    def __init__(self, data, X_vars:list, y_var:str, group_var:str=None) -> None:
        self.X_vars     = X_vars
        self.y_var      = y_var
        self.group_var  = group_var
        if group_var is None:
            self.data       = data[X_vars + [y_var]].dropna().sort_values(by=X_vars[0])
        else:
            self.data       = data[X_vars + [y_var] + [group_var]].dropna().sort_values(by=X_vars[0])
        self.X          = self.data[X_vars].values.reshape(-1,len(X_vars))
        self.y          = self.data[y_var].values.reshape(-1,1)
    
    def train_test_split(self, test_size=0.33):
        self.training, self.testing = train_test_split(self.data, test_size=test_size, random_state=42)
        self.X_training, self.X_testing, self.y_training, self.y_testing = self.training[self.X_vars], self.testing[self.X_vars], self.training[self.y_var], self.testing[self.y_var]
        if self.group_var is not None:
            self.group_training, self.group_testing = self.training[self.group_var], self.testing[self.group_var]
        return self

    @staticmethod
    def to_jax(x):
        return jnp.array(x)
    
    @staticmethod
    def extract_vars(dataset=None, vars=None):
        print('Extracting variables to JAX. DataFrame ({})'.format(dataset.shape))
        jax_vars       = {} 
        numeric_vars   = []
        cat_vars       = []
        for v in vars:
            try:
                # numeric
                numeric_vars.append(v)
                jax_vars[v] = FilterData.to_jax(dataset[v].values)
            except:
                # categorical
                cat_vars.append(v)
                jax_vars[v] = FilterData.to_jax(pd.factorize(dataset[v])[0])
        return jax_vars
    
    