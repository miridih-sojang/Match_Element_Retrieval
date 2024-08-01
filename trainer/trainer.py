class Trainer:
    def __init__(self, config):
        self.config = config

    def run(self):
        for epoch in range(self.config['train']['num_epochs']):
            self.train(epoch)
            if epoch % self.config['train']['evaluation_per_epoch'] == 0:
                self.eval(epoch)


    def train(self, epoch):
        pass

    def eval(self, epoch):
        pass