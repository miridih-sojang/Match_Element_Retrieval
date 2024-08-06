import os
import random
import yaml

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def read_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    os.environ["PYTHONHASHSEED"] = str(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def online_rendering(read_path, instance_df, data_augmentation, element_df, for_search=False):
    skin_image = instance_df[instance_df.tag == 'Skin_Images']
    element_images = instance_df[~instance_df.tag.isin(['Thumbnail_Images', 'Skin_Images'])]
    element_images.loc[:, 'priority'] = element_images['priority'].astype(float).astype(int)
    element_images = element_images.sort_values(by='priority')
    element_images.reset_index(drop=True, inplace=True)
    if for_search:
        y_candidate_idx = element_images[(~element_images.resourceKey.isin(['Not_Exists'])) &
                                         (~element_images.keywords.isin(['Unknown']))].index
    else:
        y_candidate_idx = element_images[~element_images.resourceKey.isin(['Not_Exists'])].index
    if data_augmentation == 'static-last':
        choose_y_candidate_idx = max(y_candidate_idx)
    elif data_augmentation == 'dynamic':
        choose_y_candidate_idx = np.random.choice(y_candidate_idx, size=1)[0]
    y_element_resourcekey = element_images.iloc[choose_y_candidate_idx].resourceKey
    y_element_image = element_df[element_df.resourceKey == y_element_resourcekey].iloc[0]
    x_element_images = element_images.iloc[:choose_y_candidate_idx]
    # print('************************************************************************')
    # print(y_candidate_idx, choose_y_candidate_idx)
    # print('------------------------------------------------------------------------')
    # print(x_element_images)
    # print('------------------------------------------------------------------------')
    # print(y_element_image)

    rendering_image = Image.open(f'{read_path}/{skin_image.iloc[0].save_path}/'
                                 f'{skin_image.iloc[0].reformat_image_file_name}')
    merge_image = Image.new("RGBA", rendering_image.size)
    for save_path, file_name, x_offset, y_offset, img_width, img_height in x_element_images[['save_path',
                                                                                             'reformat_image_file_name',
                                                                                             'left',
                                                                                             'top',
                                                                                             'img_width',
                                                                                             'img_height']].values:
        x_offset, y_offset, img_width, img_height = int(float(x_offset)), int(float(y_offset)), int(
            float(img_width)), int(float(img_height))
        overlay_img = Image.open(f'{read_path}/{save_path}/{file_name}').convert("RGBA")
        overlay_img = overlay_img.resize((img_width, img_height))
        _, _, _, overlay_img_mask = overlay_img.split()
        rendering_image.paste(overlay_img, (x_offset, y_offset), overlay_img_mask)
        merge_image = Image.alpha_composite(merge_image, rendering_image)
    y_image = Image.open(f'{read_path}/{y_element_image.save_path}/{y_element_image.reformat_image_file_name}')
    return merge_image, y_image, x_element_images, y_element_image
