from torch import nn
import pytorch_lightning as pl
import torchvision
import torch
import statistics
import numpy as np


class CustomSequential(nn.Module):
    def __init__(self, config):
        super(CustomSequential, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, 10))

    def forward(self, input):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).float()
        return self.model(input)

    def predict(self, input):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).float()
        return self.model(input)


class NumberNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = CustomSequential(config)
        # this is meant to operate on logits
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.test_loss = None
        self.test_accuracy = None
        self.accuracy = pl.metrics.Accuracy()
        self.training_loss_history = []
        self.avg_training_loss_history = []
        self.latest_training_loss_history = []
        self.training_loss_history = []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/resiliency/", train=True,
                                                                      transform=torchvision.transforms.ToTensor(),
                                                                      target_transform=None, download=True),
                                           batch_size=int(self.config['batch_size']), num_workers=0, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/resiliency/", train=False,
                                                                      transform=torchvision.transforms.ToTensor(),
                                                                      target_transform=None, download=True),
                                           batch_size=int(self.config['batch_size']), num_workers=0, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'],
                                     eps=self.config['adam_epsilon'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        return {'forward': self.forward(x), 'expected': y}

    def training_step_end(self, outputs):
        loss = self.criterion(outputs['forward'], outputs['expected'])
        logs = {'train_loss': loss}
        # pdb.set_trace()
        return {'loss': loss, 'logs': logs}

    def training_epoch_end(self, outputs):
        # pdb.set_trace()
        loss = []
        for x in outputs:
            loss.append(float(x['loss']))
        avg_loss = statistics.mean(loss)
        # tensorboard_logs = {'train_loss': avg_loss}
        self.avg_training_loss_history.append(avg_loss)
        self.latest_training_loss_history.append(loss[-1])
        # return {'avg_train_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        return {'forward': self.forward(x), 'expected': y}

    def test_step_end(self, outputs):
        loss = self.criterion(outputs['forward'], outputs['expected'])
        accuracy = self.accuracy(outputs['forward'], outputs['expected'])
        logs = {'test_loss': loss, 'test_accuracy': accuracy}
        return {'test_loss': loss, 'logs': logs, 'test_accuracy': accuracy}

    def test_epoch_end(self, outputs):
        loss = []
        for x in outputs:
            loss.append(float(x['test_loss']))
        avg_loss = statistics.mean(loss)
        # tensorboard_logs = {'test_loss': avg_loss}
        self.test_loss = avg_loss
        accuracy = []
        for x in outputs:
            accuracy.append(float(x['test_accuracy']))
        avg_accuracy = statistics.mean(accuracy)
        self.test_accuracy = avg_accuracy
        # return {'avg_test_loss': avg_loss, 'log': tensorboard_logs, 'avg_test_accuracy': avg_accuracy}


def mnist_pt_objective(config):
    torch.manual_seed(0)
    model = NumberNet(config)
    try:
        trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0])
    except:
        print("WARNING: training on CPU only, GPU[0] not found.")
        trainer = pl.Trainer(max_epochs=config['epochs'])
    trainer.fit(model)
    trainer.test(model)
    return (model.test_accuracy, model.model, model.avg_training_loss_history, model.latest_training_loss_history)


if __name__ == "__main__":
    # test_config = {'batch_size': 64, 'learning_rate': .001, 'epochs': 1, 'dropout': 0.5, 'adam_epsilon': 1e-7}
    test_config = {'batch_size': 800, 'learning_rate': 0.0001955, 'epochs': 15, 'dropout': 0.8869, 'adam_epsilon': 0.1182}
    res = mnist_pt_objective(test_config)
