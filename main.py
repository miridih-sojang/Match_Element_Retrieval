import argparse
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from datasets import MatchElementDataset
from utils import read_yaml, set_seed


def get_args():
    parser = argparse.ArgumentParser(description='MIRIDIH Matching Element Retrieval')
    parser.add_argument('--dataset_config_path', type=str, help='Dataset Config Path')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    config = read_yaml(args.dataset_config_path)
    set_seed(seed_num=config['seed_number'])
    # train_dataset = MatchElementDataset(config, 'train')
    test_dataset = MatchElementDataset(config, 'test')
    # train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=32, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False, drop_last=False)
    for iteration, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
        rendering_image, template_idx, page_num = batch['rendering_image'], batch['template_idx'], batch['page_num']
        rendering_image = rendering_image[0, :]
        image = Image.fromarray(rendering_image.numpy())
        image.save(f'./_visualize/{template_idx.item()}-{page_num.item()}.png')


if __name__ == '__main__':
    main()
