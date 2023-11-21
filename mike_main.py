import time
from typing import Any
import librosa
from control_data import MX20
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
import pytorch_lightning.loggers as Loggers
import pytorch_lightning.callbacks as Cb
from nvsr_unet import NVSR
import numpy as np
from voicefixer import Vocoder
#from dataset import DistanceDataModule, DAY_1_FOLDER, DAY_2_FOLDER
from dataset3 import MicroChangeDataModule
#logger = Loggers.WandbLogger(project="audio-nvsr", log_model="all")
#logger = Loggers.WandbLogger(project="micswitch", log_model="all")
logger = Loggers.CSVLogger(save_dir="logs")
model_checkpoint = Cb.ModelCheckpoint(dirpath="logs", save_top_k=-1)
trainer = L.Trainer(logger=logger, 
                     max_epochs=2, callbacks=[model_checkpoint])


if __name__ == "__main__":
    vocoder = Vocoder(sample_rate=44100)
    torch.set_float32_matmul_precision("high")

    model = NVSR(1, vocoder=vocoder)
    #datamodule = DistanceDataModule(
    #   DAY_1_FOLDER, DAY_2_FOLDER, chunk_length=32768, num_workers=24)
    datamodule = MicroChangeDataModule(
        chunk_length=32768, num_workers=24
    )

    trainer = L.Trainer(max_epochs=300, logger=logger)
    trainer.fit(model, datamodule=datamodule)
