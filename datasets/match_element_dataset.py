import numpy as np
import pandas as pd
from datasets import BaseDataset
from utils import online_rendering


class MatchElementDataset(BaseDataset):
    def __init__(self, dataset_config, training_config, data_type, processor):
        super(MatchElementDataset, self).__init__(dataset_config, data_type)
        self.training_config = training_config
        self.df = self.filter_template_page_num_by_count(self.df)
        self.element_df = self.construct_candidate_elements()
        self.element_df = self.filter_elements_by_count(self.element_df)
        self.df = self.filter_impossible_rendering(self.df)
        self.df = self.filter_not_exists_skin_images(self.df)
        self.df = self.filter_useful_rendering(self.df)
        self.grouped_df, self.filter_key_list = self.reorder_by_template()
        self.processor = processor

    def reorder_by_template(self):
        df = self.df.groupby(self.dataset_config['dataset_unique_key'])
        filter_key_list = [*df.groups.keys()]
        return df, filter_key_list

    def construct_candidate_elements(self):
        df_list = []
        for csv_name in self.dataset_config['use_csv_file_list']:
            temporal_df = pd.read_csv(f'{self.dataset_config["path"]["base_read_csv_path"]}/total/{csv_name}',
                                      low_memory=False)
            df_list.append(temporal_df)
        df = pd.concat(df_list, ignore_index=True)
        unique_elements_df = self.calculate_resolution(df)
        unique_elements_df.reset_index(inplace=True, drop=True)
        return unique_elements_df

    def __len__(self):
        return len(self.grouped_df)

    # @profile
    def __getitem__(self, idx):
        # try:
        template_idx, page_num = self.filter_key_list[idx]
        instance_df = self.df[(self.df.template_idx == template_idx) & (self.df.page_num == page_num)]
        rendering_image, y_image = online_rendering(self.dataset_config['path']['base_read_image_path'], instance_df,
                                                    self.training_config['train']['data_augmentation'])
        rendering_image, y_image = rendering_image.convert('RGB'), y_image.convert('RGB')
        # rendering_image = rendering_image.resize((500, 500))
        # y_image = y_image.resize((500, 500))

        rendering_image, y_image = self.processor(rendering_image), self.processor(y_image)
        # except:
        #     print('------------------------------------------------------------------------------------')
        #     print(template_idx, page_num, instance_df.shape, instance_df)
        #     exit()
        # return {'rendering_image': np.array(rendering_image), 'y_image': np.array(y_image),
        #         'template_idx': template_idx, 'page_num': page_num}

        return {'rendering_image': rendering_image['pixel_values'], 'y_image': y_image['pixel_values'],
                'template_idx': template_idx, 'page_num': page_num}

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
            (element_df.img_width_resized >= self.dataset_config['dataset_filter_crisis']['element']['min_width']) &
            (element_df.img_height_resized >= self.dataset_config['dataset_filter_crisis']['element']['min_height'])
            ]
        return element_df

    def filter_template_page_num_by_count(self, df):
        count_df = self.df.groupby(self.dataset_config['dataset_unique_key']).count()[
            'super_template_type'].reset_index(
            name='template_idx_page_num_count')
        use_count_df = count_df[(count_df.template_idx_page_num_count >=
                                 self.dataset_config['dataset_filter_crisis']['template']['use_element_min_count']) & (
                                        count_df.template_idx_page_num_count <=
                                        self.dataset_config['dataset_filter_crisis']['template'][
                                            'use_element_max_count'])]
        df = pd.merge(use_count_df[['template_idx', 'page_num']], df, on=['template_idx', 'page_num'], how='left')
        return df

    def filter_impossible_rendering(self, df):

        df_for_rendering = df[~df.resourceKey.isin(['Skin_Images', 'Thumbnail_Images'])]
        df_for_rendering.loc[:, 'img_width'] = df_for_rendering['img_width'].astype(float).astype(int)
        df_for_rendering.loc[:, 'img_height'] = df_for_rendering['img_height'].astype(float).astype(int)

        impossible_rendering_df = \
            df_for_rendering[
                (df_for_rendering.img_width < self.dataset_config['dataset_filter_crisis']['rendering'][
                    'min_width']) | (
                        df_for_rendering.img_height < self.dataset_config['dataset_filter_crisis']['rendering'][
                    'min_height'])][
                ['template_idx', 'page_num']].drop_duplicates().reset_index(drop=True)

        impossible_rendering_df['name'] = impossible_rendering_df['template_idx'].astype(str) + '**' + \
                                          impossible_rendering_df['page_num'].astype(str)
        df['name'] = df['template_idx'].astype(str) + '**' + df['page_num'].astype(str)
        df = df[~df.name.isin(impossible_rendering_df.name)]
        df.drop(columns=['name'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    def filter_not_exists_skin_images(self, df):
        not_exists_skin_images_list = []
        for name, instance_df in df.groupby(self.dataset_config['dataset_unique_key']):
            filtering_df = instance_df[instance_df.reformat_image_file_name.str.endswith('_1.png')]
            template_idx, page_num = name
            if filtering_df.shape[0] == 0:
                not_exists_skin_images_list.append(f'{template_idx}**{page_num}')
        df['name'] = df['template_idx'].astype(str) + '**' + df['page_num'].astype(str)
        df = df[~df.name.isin(not_exists_skin_images_list)]
        df.drop(columns=['name'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    def filter_useful_rendering(self, df):
        not_useful_images_list = []
        for name, instance_df in df.groupby(self.dataset_config['dataset_unique_key']):
            filtering_df = instance_df[~instance_df.tag.isin(['Thumbnail_Images', 'Skin_Images'])]
            template_idx, page_num = name
            filtering_df.loc[:, 'priority'] = filtering_df['priority'].astype(float).astype(int)
            filtering_df = filtering_df.sort_values(by='priority')
            filtering_df.reset_index(drop=True, inplace=True)
            y_candidate_idx = filtering_df[~filtering_df.resourceKey.isin(['Not_Exists'])].index
            if y_candidate_idx.shape[0] == 0:
                not_useful_images_list.append(f'{template_idx}**{page_num}')
            elif min(y_candidate_idx) < 1:
                not_useful_images_list.append(f'{template_idx}**{page_num}')
        df['name'] = df['template_idx'].astype(str) + '**' + df['page_num'].astype(str)
        df = df[~df.name.isin(not_useful_images_list)]
        return df
