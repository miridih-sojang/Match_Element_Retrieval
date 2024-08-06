import pandas as pd

from utils import online_rendering


class KeywordBasedSearch:
    def __init__(self, dataset_config, df, element_df):
        self.dataset_config = dataset_config
        self.df = df
        self.element_df = element_df
        self.vocabulary_df = self.construct_retrieval_vocabulary()
        print(self.vocabulary_df)
        exit()

    def construct_retrieval_vocabulary(self):
        vocabulary = []

        for name, instance_df in self.df.groupby(self.dataset_config['dataset_unique_key']):
            x_image, y_image, x_info, y_info = online_rendering(self.dataset_config['path']['base_read_image_path'],
                                                                instance_df, 'static-last', self.element_df,
                                                                for_search=True)
            template_idx, page_num = name
            for search_word in y_info['keywords']:
                vocabulary.append(
                    [search_word, x_image, y_image, template_idx, page_num, x_info.to_dict(), y_info.to_dict(),
                     y_info.save_path, y_info.resourceKey, y_info.reformat_image_file_name])
        vocabulary_df = pd.DataFrame(vocabulary, columns=['search_word', 'x_image', 'y_image', 'template_idx',
                                                          'page_num', 'x_info', 'y_info', 'y_image_save_path',
                                                          'resourceKey', 'y_info_reformat_image_file_name'])
        vocabulary_df.to_csv('./vocabulary.csv', index=False)
        return vocabulary_df
# Keyword <-> Image ( Removed last Rendering )
