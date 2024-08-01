import os
import random
import yaml

import numpy as np
import torch
from PIL import Image


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


def online_rendering(read_path, instance_df):
    skin_image = instance_df[instance_df.tag == 'Skin_Images']
    element_images = instance_df[~instance_df.tag.isin(['Thumbnail_Images', 'Skin_Images'])]
    element_images.loc[:, 'priority'] = element_images['priority'].astype(float).astype(int)
    element_images = element_images.sort_values(by='priority')
    rendering_image = Image.open(f'{read_path}/{skin_image.iloc[0].save_path}/'
                                 f'{skin_image.iloc[0].reformat_image_file_name}')
    merge_image = Image.new("RGBA", rendering_image.size)
    for save_path, file_name, x_offset, y_offset, img_width, img_height in element_images[['save_path',
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
    return merge_image
