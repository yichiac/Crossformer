import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
# import geopandas as gpd
import os

class CropYieldDataset(Dataset):
    """
    Dataset for loading crop yield data, including static (soil) and temporal (weather, satellite) features.

    Args:
        df (pd.DataFrame): Dataframe containing the dataset.
        years (list): List of years to include in the dataset.
        static_cols (list): List of columns corresponding to static features.
        temporal_cols (list): List of columns corresponding to temporal features.
        target_col (list): List of columns corresponding to the target (crop yield).
        sequence_doys (list): List of days of year for temporal sequences.
    """
    def __init__(self, df, years, static_cols, temporal_cols, target_col, meta_cols, sequence_doys):
        self.df = df[df['YEAR'].isin(years)]
        self.static_cols = static_cols
        self.temporal_cols = temporal_cols
        self.target_col = target_col
        self.meta_cols = meta_cols
        self.sequence_length = len(sequence_doys)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # get temporal features, reshape and permute them as needed
        x_temporal = torch.tensor(row[self.temporal_cols].values, dtype=torch.float32).view(-1, self.sequence_length).permute(-1, -2)
        x_static = torch.tensor(row[self.static_cols].values, dtype=torch.float32)
        # x_meta = row[self.meta_cols].values
        y = torch.tensor(row[self.target_col].values, dtype=torch.float32)

        return x_temporal, x_static, y


class CropYieldDataModule(pl.LightningDataModule):
    """
    Data module for handling loading of crop yield data, including cross-validation or train-test splits.

    Args:
        data_path (str): Path to the pickle file containing the dataset.
        batch_size (int): Batch size for data loaders. Default is 32.
        repeat_static (bool): Whether to repeat static features across the temporal sequence. Default is False.
        fold (int, optional): Fold number for cross-validation. If None, performs train-test split. Default is None.
        train_years (list): List of years to use for training. Default is None (uses cross-validation years).
        val_years (list): List of years to use for validation. Default is None (uses cross-validation years).
        test_years (list): List of years to use for testing. Default is [2018, 2019, 2020, 2021].
    """
    def __init__(self, data_path, batch_size=32, repeat_static=False, fold=None,
                 train_years=None, val_years=None, test_years=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.repeat_static = repeat_static
        self.fold = fold
        self.train_years = train_years if train_years is not None else list(range(2001, 2018))
        self.val_years = val_years
        self.test_years = test_years if test_years is not None else list(range(2018, 2022))

    # def extract_latlon(self, df):
    #     shp_path = os.path.join(os.path.dirname(self.data_path), 'BR_Municipios_2022', 'BR_Municipios_2022.shp')
    #     shp_mun = gpd.read_file(shp_path, engine='pyogrio')
    #     shp_mun['CD_MUN'] = shp_mun['CD_MUN'].astype(int)  # city code
    #     shp_mun['STATEFP'] = shp_mun['CD_MUN']//100000     # state code
    #     shp_mun = shp_mun[['CD_MUN','SIGLA_UF','STATEFP','geometry']]
    #     shp_mun['LAT'] = shp_mun.centroid.y
    #     shp_mun['LON'] = shp_mun.centroid.x
    #     centroids = shp_mun[['CD_MUN', 'LAT', 'LON']]
    #     df = pd.merge(df, centroids, on='CD_MUN', how='left')
    #     return df

    def normalization(self, df):
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

        for var, prefix in variable_groups.items():
            cols = [col for col in df.columns if prefix in col]
            df = min_max_normalize(df, cols)

        df['YIELD'] = (df['YIELD'] - df['YIELD'].min()) / (df['YIELD'].max() - df['YIELD'].min())
        for static_col in self.static_cols:
            df[static_col] = (df[static_col] - df[static_col].min()) / (df[static_col].max() - df[static_col].min())
        return df

    def prepare_data(self):
        self.df = pd.read_pickle(self.data_path) # load dataset from pickle file
        # self.df = self.extract_latlon(self.df)
        self.meta_vars = ['CD_MUN', 'YEAR', 'YIELD', 'LAT', 'LON']
        self.soil_vars = ['om', 'clay', 'sand', 'theta_r', 'theta_s']
        self.weather_vars = ['ppt', 'tmax', 'tmin', 'vpdmax', 'vpdmin']
        self.satellite_vars = ['green', 'nir']
        self.sequence_doys = range(-90, 150)

        self.static_cols = [f'soil/{_var}/mean/static' for _var in self.soil_vars]
        self.temporal_cols = [
            f'mswx/{_var}/mean/doy_{_doy}' for _var in self.weather_vars for _doy in self.sequence_doys
        ] + [
            f'satellite/{_var}/mean/doy_{_doy}' for _var in self.satellite_vars for _doy in self.sequence_doys
        ]
        # self.meta_cols = self.meta_vars
        self.target_col = ['YIELD']
        self.df = self.normalization(self.df)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.fold is not None:
                # k-fold siding window cross-validation
                num_years = len(self.train_years)
                self.train_datasets = []
                self.val_datasets = []
                for fold in range(5):
                    start_idx = fold
                    end_idx = num_years - 5 + fold - 1
                    val_idx = end_idx + 1

                    train_years = self.train_years[start_idx:end_idx + 1]
                    val_years = [self.train_years[val_idx]]

                    print(f"Fold k = {fold}:")
                    print(f"train years: {train_years}")
                    print(f"val years: {val_years}")

                    self.train_datasets.append(CropYieldDataset(self.df, train_years, self.static_cols, self.temporal_cols, self.target_col, self.meta_vars, self.sequence_doys))
                    self.val_datasets.append(CropYieldDataset(self.df, val_years, self.static_cols, self.temporal_cols, self.target_col, self.meta_vars, self.sequence_doys))
            else:
                # standard train-test split
                self.train_dataset = CropYieldDataset(self.df, self.train_years, self.static_cols, self.temporal_cols, self.target_col, self.meta_vars, self.sequence_doys)
                if self.val_years:
                    self.val_dataset = CropYieldDataset(self.df, self.val_years, self.static_cols, self.temporal_cols, self.target_col, self.meta_vars, self.sequence_doys)

        # setup for testing
        if stage == 'test' or stage is None:
            self.test_dataset = CropYieldDataset(self.df, self.test_years, self.static_cols, self.temporal_cols, self.target_col, self.meta_vars, self.sequence_doys)

    def train_dataloader(self):
        if self.fold is not None:
            return DataLoader(self.train_datasets[self.fold-1], batch_size=self.batch_size, shuffle=False)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        if self.fold is not None:
            return DataLoader(self.val_datasets[self.fold-1], batch_size=self.batch_size, shuffle=False)
        elif self.val_years:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            return None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
