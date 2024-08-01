import numpy as np
import pandas as pd
from datasets import BaseDataset
from utils import online_rendering


class MatchElementDataset(BaseDataset):
    def __init__(self, config, data_type):
        super(MatchElementDataset, self).__init__(config, data_type)
        self.df = self.filter_template_page_num_by_count(self.df)
        self.grouped_df, self.filter_key_list = self.reorder_by_template()
        self.element_df = self.construct_candidate_elements()
        self.element_df = self.filter_elements_by_count(self.element_df)
        if self.data_type == 'test':
            self.df = self.filter_test_df(self.df)

    def reorder_by_template(self):
        df = self.df.groupby(self.config['dataset_unique_key'])
        filter_key_list = [*df.groups.keys()]
        return df, filter_key_list

    def construct_candidate_elements(self):
        df_list = []
        for csv_name in self.config['use_csv_file_list']:
            temporal_df = pd.read_csv(f'{self.config["path"]["base_read_csv_path"]}/total/{csv_name}', low_memory=False)
            df_list.append(temporal_df)
        df = pd.concat(df_list, ignore_index=True)
        unique_elements_df = self.calculate_resolution(df)
        unique_elements_df.reset_index(inplace=True, drop=True)
        return unique_elements_df

    def __len__(self):
        return len(self.grouped_df)

    def __getitem__(self, idx):
        template_idx, page_num = self.filter_key_list[idx]
        instance_df = self.df[(self.df.template_idx == template_idx) & (self.df.page_num == page_num)]
        rendering_image = online_rendering(self.config['path']['base_read_image_path'], instance_df)
        return {'rendering_image': np.array(rendering_image), 'template_idx': template_idx, 'page_num': page_num}

    @staticmethod
    def calculate_resolution(df):
        elements_df = df[~df.resourceKey.isin(['Not_Exists', 'Skin_Images', 'Thumbnail_Images'])]
        elements_df.loc[:, 'img_width'] = elements_df['img_width'].astype(float).astype(int)
        elements_df.loc[:, 'img_height'] = elements_df['img_height'].astype(float).astype(int)
        elements_df.loc[:, 'resolution'] = elements_df['img_width_resized'] * elements_df['img_height_resized']
        max_resolution_by_resourcekey = elements_df.groupby('resourceKey')['resolution'].idxmax()
        unique_elements_df = elements_df.loc[max_resolution_by_resourcekey]
        return unique_elements_df

    def filter_elements_by_count(self, element_df):
        element_df = element_df[
            (element_df.img_width_resized >= self.config['dataset_filter_crisis']['element']['min_width']) &
            (element_df.img_height_resized >= self.config['dataset_filter_crisis']['element']['min_height'])
            ]
        return element_df

    def filter_template_page_num_by_count(self, df):
        count_df = self.df.groupby(self.config['dataset_unique_key']).count()['super_template_type'].reset_index(
            name='template_idx_page_num_count')
        use_count_df = count_df[(count_df.template_idx_page_num_count >=
                                 self.config['dataset_filter_crisis']['template']['use_element_min_count']) & (
                                        count_df.template_idx_page_num_count <=
                                        self.config['dataset_filter_crisis']['template']['use_element_max_count'])]
        df = pd.merge(use_count_df[['template_idx', 'page_num']], df, on=['template_idx', 'page_num'], how='left')
        return df

    def filter_test_df(self, df):

        df_for_rendering = df[~df.resourceKey.isin(['Skin_Images', 'Thumbnail_Images'])]
        df_for_rendering.loc[:, 'img_width'] = df_for_rendering['img_width'].astype(float).astype(int)
        df_for_rendering.loc[:, 'img_height'] = df_for_rendering['img_height'].astype(float).astype(int)

        impossible_rendering_df = \
            df_for_rendering[
                (df_for_rendering.img_width < self.config['dataset_filter_crisis']['rendering']['min_width']) | (
                        df_for_rendering.img_height < self.config['dataset_filter_crisis']['rendering']['min_height'])][
                ['template_idx', 'page_num']].drop_duplicates().reset_index(drop=True)

        impossible_rendering_df['name'] = impossible_rendering_df['template_idx'].astype(str) + '**' + \
                                          impossible_rendering_df['page_num'].astype(str)
        df['name'] = df['template_idx'].astype(str) + '**' + df['page_num'].astype(str)
        df = df[~df.name.isin(impossible_rendering_df.name)]
        df.drop(columns=['name'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df
