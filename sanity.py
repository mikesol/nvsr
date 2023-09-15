import time
from typing import Any
import librosa
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as L
import pytorch_lightning.loggers as Loggers
import pytorch_lightning.callbacks as Cb
from nvsr_unet import NVSR
import numpy as np
from dataset import DistanceDataModule, DAY_1_FOLDER, DAY_2_FOLDER


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model = NVSR(1)
    data_module = DistanceDataModule(
        DAY_1_FOLDER, DAY_2_FOLDER, chunk_length=32768, num_workers=24
    )

    data_module.setup(stage='validate')  # sets up datasets

    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    inputs, targets = batch
    torchaudio.save("sanity_input_val.wav", inputs[0], sample_rate=44100)
    torchaudio.save("sanity_output_val.wav", targets[0], sample_rate=44100)