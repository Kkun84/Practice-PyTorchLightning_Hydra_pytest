from logging import getLogger
import hydra
import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

from . import network


logger = getLogger(__name__)


class Model(network.Discriminator):

    def __init__(self, model_params, hparams, optim, data_path):
        super().__init__(model_params)
        self.hparams = argparse.Namespace(**hparams)
        self.optim = optim
        self.data_path = data_path

    def configure_optimizers(self):
        logger.debug('configure_optimizers')
        optim = hydra.utils.instantiate(self.optim.optimizer, self.parameters())
        if self.optim.scheduler:
            scheduler = hydra.utils.instantiate(self.optim.scheduler, optim)
            return [optim], [scheduler]
        return [optim], []

    def prepare_data(self):
        logger.debug('prepare_data')
        train_dataset = MNIST(root=self.data_path, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = MNIST(root=self.data_path, train=False, download=True, transform=transforms.ToTensor())

        train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        logger.debug('train_dataloader')
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        return DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def val_dataloader(self):
        logger.debug('val_dataloader')
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers)

    def test_dataloader(self):
        logger.debug('test_dataloader')
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        return DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers)

    def training_step(self, batch, batch_idx):
        logger.debug(f'training_step-{batch_idx}')
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        logger.debug(f'validation_step-{batch_idx}')
        x, y = batch
        y_hat = self(x)
        correct = (y == y_hat.argmax(1)).float()
        return {'loss': F.cross_entropy(y_hat, y, reduction='sum'), 'correct': correct}

    def validation_epoch_end(self, outputs):
        logger.debug('validation_epoch_end')
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum() / len(self.val_dataset)
        accuracy = torch.cat([x['correct'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': accuracy}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        logger.debug(f'test_step-{batch_idx}')
        x, y = batch
        y_hat = self(x)
        correct = (y == y_hat.argmax(1)).float()
        return {'loss': F.cross_entropy(y_hat, y, reduction='sum'), 'correct': correct}

    def test_epoch_end(self, outputs):
        logger.debug('test_epoch_end')
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum() / len(self.test_dataset)
        accuracy = torch.cat([x['correct'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_accuracy': accuracy}
        if self.logger is not None:
            self.logger.log_metrics(tensorboard_logs, self.global_step)
        return {'test_loss': avg_loss, 'log': tensorboard_logs}
