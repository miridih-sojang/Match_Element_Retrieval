import pandas as pd
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config, data_type):
        self.dataset_config = dataset_config
        self.data_type = data_type
        self.df = self.read_csv()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def read_csv(self):
        df_list = []
        for csv_name in self.dataset_config['use_csv_file_list']:
            temporal_df = pd.read_csv(f'{self.dataset_config["path"]["base_read_csv_path"]}/{self.data_type}/{csv_name}',
                                      low_memory=False)
            df_list.append(temporal_df)
        df = pd.concat(df_list, ignore_index=True)
        df = df[self.dataset_config['use_columns']]
        return df
