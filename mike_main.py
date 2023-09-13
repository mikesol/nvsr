import time
from typing import Any
import librosa
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
import pytorch_lightning.loggers as Loggers
import pytorch_lightning.callbacks as Cb
from nvsr_unet import NVSR
import numpy as np
from dataset import DistanceDataModule, DAY_1_FOLDER, DAY_2_FOLDER
logger = Loggers.WandbLogger(project="mic-nvsr-transfer", log_model="all")
model_checkpoint = Cb.ModelCheckpoint(dirpath="logs", save_top_k=-1)
trainer = L.Trainer(logger=logger, max_epochs=2, callbacks=[model_checkpoint])


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    state_dict = torch.load('../nvsr.ckpt')
    model = NVSR(1, vocoder=True) # NVSR.load_from_checkpoint('../nvsr.ckpt', channels=1)
    model.load_state_dict(state_dict)
    datamodule = DistanceDataModule(
        DAY_1_FOLDER, DAY_2_FOLDER, chunk_length=32768, num_workers=24
    )

    trainer = L.Trainer(max_epochs=300, logger=logger)
    trainer.fit(model, datamodule=datamodule)
