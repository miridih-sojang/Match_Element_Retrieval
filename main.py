import argparse
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from transformers.models.efficientnet.modeling_efficientnet import EfficientNetConfig

from datasets import MatchElementDataset
from trainer import Trainer
from utils import read_yaml, set_seed
from models import EfficientNetDualModel


def get_args():
    parser = argparse.ArgumentParser(description='MIRIDIH Matching Element Retrieval')
    parser.add_argument('--dataset_config_path', type=str, help='Dataset Config Path')
    parser.add_argument('--training_config_path', type=str, help='Training Config Path')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset_config, training_config = read_yaml(args.dataset_config_path), read_yaml(args.training_config_path)
    model_config = read_yaml(f'./config/models/{training_config["train"]["model_name"]}.yaml')

    set_seed(seed_num=training_config['seed_number'])

    if training_config["train"]["model_name"] == 'Efficientnet':
        processor = AutoImageProcessor.from_pretrained(
            model_config['model_name_or_path'],
            cache_dir=model_config['cache_dir'],
            revision=model_config['model_revision'],
            token=model_config['token'],
            trust_remote_code=model_config['trust_remote_code']
        )
        config = EfficientNetConfig.from_pretrained(model_config['model_name_or_path'])
        config.projection_dim = config.hidden_dim
        config.logit_scale_init_value = model_config['logit_scale_init_value']
        model = EfficientNetDualModel.from_pretrained(
            model_config['model_name_or_path'],
            config=config,
            cache_dir=model_config['cache_dir'],
            revision=model_config['model_revision'],
            token=model_config['token'],
            trust_remote_code=model_config['trust_remote_code'],
        )

    train_dataset = MatchElementDataset(dataset_config, training_config, 'train', processor)
    test_dataset = MatchElementDataset(dataset_config, training_config, 'test', processor)

    train_dataloader = DataLoader(train_dataset, batch_size=training_config['train']['batch_size'],
                                  num_workers=training_config['train']['batch_size'], shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=training_config['test']['batch_size'],
                                 num_workers=training_config['test']['batch_size'], shuffle=False, drop_last=False)

    # from PIL import Image
    # for iteration, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
    #     rendering_image, template_idx, page_num, y_image = batch['rendering_image'], batch['template_idx'], batch['page_num'], batch['y_image']
    #     rendering_image = rendering_image[0, :]
    #     image = Image.fromarray(rendering_image.numpy())
    #     image.save(f'./_visualize/{template_idx.item()}-{page_num.item()}-0.png')
    #     y_image = y_image[0, :]
    #     image = Image.fromarray(y_image.numpy())
    #     image.save(f'./_visualize/{template_idx.item()}-{page_num.item()}-1.png')

    trainer = Trainer(config=training_config,
                      model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      evaluator=None)
    trainer.run()





if __name__ == '__main__':
    main()
