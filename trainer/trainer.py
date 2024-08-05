from tqdm import tqdm

import torch

from losses import ClipLoss


class Trainer:
    def __init__(self,
                 config,
                 model,
                 train_dataloader,
                 test_dataloader,
                 evaluator):
        self.config = config
        self.model = model
        self.train_dataloader, self.test_dataloader = train_dataloader, test_dataloader
        self.evaluator = evaluator
        self.loss, self.optimizer = self.get_loss(), self.get_optimizer()

    def run(self):
        for epoch in range(self.config['train']['num_epochs']):
            self.train(epoch)
            if epoch % self.config['test']['per_epoch'] == 0:
                self.test(epoch)

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            element_input, background_input = batch['y_image'][0], batch['rendering_image'][0]
            outputs = self.model(element_input, background_input)
            running_loss = self.loss(outputs)
            running_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += running_loss.item()
        print(f'**** Train {epoch} Epoch --> Loss : {epoch_loss / len(self.train_dataloader)} ****')

    @torch.no_grad()
    def test(self, epoch):
        epoch_loss = 0.0
        for batch in tqdm(self.test_dataloader, total=len(self.test_dataloader)):
            element_input, background_input = batch['y_image'][0], batch['rendering_image'][0]
            outputs = self.model(element_input, background_input)
            epoch_loss += self.loss(outputs)
        print(f'**** Test {epoch} Epoch --> Loss : {epoch_loss / len(self.test_dataloader)} ****')

    def get_loss(self):
        if self.config['train']['loss'] == 'Cross_Entropy':
            return torch.nn.CrossEntropyLoss()
        elif self.config['train']['loss'] == 'Clip_Loss':
            return ClipLoss()

        else:
            raise NotImplementedError

    def get_optimizer(self):
        if self.config['train']['optimizer']['name'] == 'Adam':
            return torch.optim.Adam(self.model.parameters(),
                                    lr=float(self.config['train']['optimizer']['learning_rate']),
                                    betas=eval(self.config['train']['optimizer']['betas']),
                                    eps=float(self.config['train']['optimizer']['eps']),
                                    weight_decay=float(self.config['train']['optimizer']['weight_decay']))
        else:
            raise NotImplementedError
