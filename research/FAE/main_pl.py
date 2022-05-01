# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py.

Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.

Train on ImageNet with default parameters:

.. code-block: bash

    python imagenet.py fit --model.data_path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python imagenet.py --help
    python imagenet.py fit --help
"""
import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
# import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.utilities.cli import LightningCLI

import models
from celeba import CelebA

class CelebALightningModel(LightningModule):
    """
    >>> CelebALightningModel(data_path='missing')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    CelebALightningModel(
      (model): ResNet(...)
    )
    """

    def __init__(
        self,
        data_path: str,
        arch: str = "resnet50",
        pretrained: bool = False,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        workers: int = 4,
    ):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.model = models.__dict__[self.arch](pretrained=self.pretrained)
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.train_acc1 = Accuracy(top_k=1)
        # self.train_acc5 = Accuracy(top_k=5)
        self.eval_acc1 = Accuracy(top_k=1)
        # self.eval_acc5 = Accuracy(top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)

        # loss_train = F.cross_entropy(output, target)
        loss_train = 0
        acc1 = 0
        for j in range(len(output)):
            loss_train += F.cross_entropy(output[j], target[:, j])
            acc1_tm = self.train_acc1(output[j], target[:, j])
            acc1 += acc1_tm / len(output)

        self.log("train_loss", loss_train)
        # update metrics
        # self.train_acc1(output, target)

        self.log("train_acc1", acc1, prog_bar=True)
        # tensorboard_logs = {"train_loss": loss_train, "train_acc1": acc1}
        # self.log("train_performance", tensorboard_logs)
        return loss_train
        # return {"loss": loss_train}

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        # loss_val = F.cross_entropy(output, target)

        loss_val = 0
        acc1 = 0
        for j in range(len(output)):
            loss_val += F.cross_entropy(output[j], target[:, j])
            acc1_tm = self.eval_acc1(output[j], target[:, j])
            acc1 += acc1_tm / len(output)
            
        self.log(f"{prefix}_loss", loss_val)
        # update metrics
        # self.eval_acc1(output, target)
        # self.eval_acc5(output, target)
        self.log(f"{prefix}_acc1", acc1, prog_bar=True)
        # self.log(f"{prefix}_acc5", self.eval_acc5, prog_bar=True)

        # tensorboard_logs = {f"{prefix}_loss": loss_val, f"{prefix}_acc1": acc1}

        return loss_val
        # return {"loss": loss_val, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None):
        if isinstance(self.trainer.strategy, ParallelStrategy):
            # When using a single GPU per process and per `DistributedDataParallel`, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        if stage in (None, "fit"):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_dataset = CelebA(
                self.data_path,
                'train_attr_list.txt',
                transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        # all stages will use the eval dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # val_dir = os.path.join(self.data_path, "val")
        # self.eval_dataset = datasets.ImageFolder(
        #     val_dir,
        #     transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]),
        # )
        self.eval_dataset = CelebA(self.data_path, 'val_attr_list.txt', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
        self.test_dataset = CelebA(self.data_path, 'test_attr_list.txt', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))       
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )


if __name__ == "__main__":


    LightningCLI(
        CelebALightningModel,
        trainer_defaults={
            "max_epochs": 90,
            "accelerator": "auto",
            "devices": 1,
            "logger": True,
            "benchmark": True,
            "callbacks": [
                # the PyTorch example refreshes every 10 batches
                TQDMProgressBar(refresh_rate=10),
                # save when the validation top1 accuracy improves
                ModelCheckpoint(monitor="val_acc1", mode="max", save_top_k=5),
            ],
        },
        seed_everything_default=42,
        save_config_overwrite=True,
    )