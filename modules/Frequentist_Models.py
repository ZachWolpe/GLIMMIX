
from modules.dependencies import *

class FrequentistModels:

    models = {
        'Linear Regression': LinearRegression,
        'Linear GAM':        LinearGAM
    }

    def __init__(self, X_train, y_train, X_test, y_test, model_name='Linear Regression') -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.model   = self.models[model_name]().fit(self.X_train, self.y_train)
        self.model_name = model_name


    def set_model(self, model_name):
        assert model_name in FrequentistModels.models.keys(), 'Model must be one of [{}]'.format(FrequentistModels.models.keys())
        self.model = model_name
        return self
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        return self

    def predict(self, X=None):
        if X is not None:
            return self.model.predict(X)
        self.y_pred = self.model.predict(self.X_test)
        return self
    
    def print_model(self):
        if hasattr(self.model, 'summary'):
            print(self.model.summary())
        else:
            print(self.model)
        return self

    def print_score(self):
        print('R^2: ', r2_score(self.y_test, self.y_pred))
        return self
    


    def plot_model(self, fd, c='orange', c2='lightblue', x_index=0):
        self.theme = 'plotly_dark'
        self.theme = 'none'

        X_vars = fd.X_vars
        y_var  = fd.y_var

        if self.model_name=='Linear GAM':
            name  = 'y ~ GAM(X)'
            title = 'Generalized Additive Model'
        else:
            name = 'y ~ f(X)'
            title = 'Linear Regression'


        def generate_plotting_df(fd):
            plot_df         = fd.training.copy()
            plot_df['set']  = 'training'
            df2             = fd.testing.copy()
            df2['set']      = 'testing'
            return pd.concat([plot_df, df2])
        
        # Generate DataFrame
        plot_df             = generate_plotting_df(fd)
        plot_df['y_hat']    = self.predict(plot_df[X_vars].values)
        plot_df             = plot_df.sort_values(by=X_vars[0])

        # plot 

        fig = go.Figure()
        for st in ['training', 'testing']:
            sub = plot_df[plot_df['set']==st]
            if st == 'training':
                opacity = 0.3
                n1 = 'training'
            else:
                opacity = 1
                n1 = 'testing'
            fig.add_trace(go.Line(x=sub[X_vars].iloc[:,x_index], y=sub[y_var], mode='markers', marker_color=c, name=n1, opacity=opacity))

        fig.add_trace(go.Line(x=plot_df[X_vars].iloc[:,x_index], y=plot_df['y_hat'], mode='lines+markers', marker_color=c2, name=name, opacity=opacity))
        fig.update_layout(template=self.theme, title=title)
        return fig
    