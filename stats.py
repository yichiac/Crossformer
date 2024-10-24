import pandas as pd
import numpy as np
import os

base_path = 'datasets'
data_path = os.path.join(base_path, 'brazil_soybeans/Brazil_soybeans_data.pkl')

df = pd.read_pickle(data_path) # load dataset from pickle file
# self.df = self.extract_latlon(self.df)
meta_vars = ['CD_MUN', 'YEAR', 'YIELD', 'LAT', 'LON']
soil_vars = ['om', 'clay', 'sand', 'theta_r', 'theta_s']
weather_vars = ['ppt', 'tmax', 'tmin', 'vpdmax', 'vpdmin']
satellite_vars = ['green', 'nir']
sequence_doys = range(-90, 150)

static_cols = [f'soil/{_var}/mean/static' for _var in soil_vars]
temporal_cols = [
    f'mswx/{_var}/mean/doy_{_doy}' for _var in weather_vars for _doy in sequence_doys
] + [
    f'satellite/{_var}/mean/doy_{_doy}' for _var in satellite_vars for _doy in sequence_doys
]
# meta_cols = meta_vars
target_col = ['YIELD']

def normalization(df):
    def min_max_normalize(df, columns):
        min_val = df[columns].stack().min()
        max_val = df[columns].stack().max()
        df[columns] = (df[columns] - min_val) / (max_val - min_val)
        return df

    variable_groups = {
        'ppt': 'mswx/ppt/mean/doy_',
        'tmax': 'mswx/tmax/mean/doy_',
        'tmin': 'mswx/tmin/mean/doy_',
        'vpdmax': 'mswx/vpdmax/mean/doy_',
        'vpdmin': 'mswx/vpdmin/mean/doy_',
        'green': 'satellite/green/mean/doy_',
        'nir': 'satellite/nir/mean/doy_',
    }

    # Normalize all variables based on their column prefixes
    for var, prefix in variable_groups.items():
        cols = [col for col in df.columns if prefix in col]
        df = min_max_normalize(df, cols)

    df['YIELD'] = (df['YIELD'] - df['YIELD'].min()) / (df['YIELD'].max() - df['YIELD'].min())
    # soil_vars = ['om', 'clay', 'sand', 'theta_r', 'theta_s']
    # static_cols = [f'soil/{_var}/mean/static' for _var in soil_vars]
    # for static_col in static_cols:
    for static_col in self.static_cols:
        df[static_col] = (df[static_col] - df[static_col].min()) / (df[static_col].max() - df[static_col].min())
    return df

df2 = normalization(df)

print('done')
