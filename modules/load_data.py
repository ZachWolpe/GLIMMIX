
from modules.dependencies import *

class CountriesData:

    def __init__(self, path='./data/countries of the world.csv') -> None:
        countries = pd.read_csv(path)
        
        # encode Region
        vals, region_map = pd.factorize(countries['Region'])
        region_map = {v:r for r,v in zip(region_map, vals)}
        countries['Region_Encode'] = vals

        # clean
        countries.iloc[:,:2] = countries.iloc[:,:2].applymap(lambda x: str(x).strip())
        countries.iloc[:,2:] = countries.iloc[:,2:].applymap(lambda x: str(x).replace(',', '.')).apply(pd.to_numeric, errors='coerce') 

        # store
        self.region_map = region_map
        self.countries  = countries

    
    def correlations(self, col='GDP ($ per capita)', df=None):
        if df is None:
            df = self.countries.iloc[:,2:]
        if col is None:
            return df.corr()
        return df.corr()[col].sort_values(ascending=False)


CD = CountriesData()

class CountryPlots:
    template    = 'plotly_dark'
    template    = 'none'
    countries   = CD.countries

    @staticmethod
    def violin(data_frame, x='Region', y="GDP ($ per capita)", color="Region", title='GDP ($ per capita) per Region', *args, **kwargs):
        fig = px.violin(data_frame=data_frame, y=y, x=x, color=color, box=True, points="all", hover_data=CountryPlots.countries.columns, *args, **kwargs)
        fig.update_layout(title=title, template=CountryPlots.template)
        return fig
    
    @staticmethod
    def scatter(data_frame, x='Phones (per 1000)', y='GDP ($ per capita)', color='Region', *args, **kwargs):
        fig = px.scatter(data_frame=data_frame, x=x, y=y, color=color, *args, **kwargs)
        fig.update_layout(template=CountryPlots.template, title='')
        return fig
    
    @staticmethod
    def scatter_3d(data_frame, x='Phones (per 1000)', y='GDP ($ per capita)', z='Literacy (%)', color='Region', *args, **kwargs):
        fig = px.scatter_3d(data_frame=data_frame, x=x, y=y, z=z, color=color, *args, **kwargs)
        fig.update_layout(template=CountryPlots.template, title='')
        return fig
