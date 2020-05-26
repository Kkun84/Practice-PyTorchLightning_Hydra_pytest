from logging import getLogger
import torch
from torch.nn import functional as F
import pytorch_lightning as pl


logger = getLogger(__name__)


class Discriminator(pl.LightningModule):

    input_shape = (28, 28)
    output_shape = (10,)

    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        self.hparams = None

        hidden_dim = self.model_params.hidden_dim
        self.linear0 = torch.nn.Linear(28 * 28, hidden_dim)
        self.linear1 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        logger.debug(f"input-x.shape={x.shape}")

        x = x.reshape(len(x), 28**2)
        logger.debug(f"reshaped-x.shape={x.shape}")

        x = self.linear0(x)
        logger.debug(f"linear0-x.shape={x.shape}")

        x = F.relu(x)

        x = self.linear1(x)
        logger.debug(f"linear1-x.shape={x.shape}")

        logger.debug(f"output-x.shape={x.shape}")
        return x
