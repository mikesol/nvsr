from control_data import MX20, RAW_SAMPLES_DAY_1, RAW_SAMPLES_DAY_2, RAW_COMPRESSOR_DAY_1, RAW_COMPRESSOR_DAY_2
from collections import defaultdict
from itertools import tee
from pathlib import Path
import pytorch_lightning as pl
import re
import sys
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

class RecordingDataset(Dataset):
    def __init__(
        self,
        mx20: MX20,
        chunk_length: int = 2048,
    ):
        self.mx20 = mx20
        self.chunk_length = chunk_length
        self.num_frames = torchaudio.info(f'o_x_{mx20}.wav').num_frames

    def __getitem__(self, marker: int):
        
        with sf.SoundFile(f'o_x_{self.mx20}.wav', "r") as f:
            frame_index = self.chunk_length * marker
            f.seek(frame_index)
            input_audio = f.read(self.chunk_length, dtype="float32")

        with sf.SoundFile(f'o_y_{self.mx20}.wav', "r") as f:
            frame_index = self.chunk_length * marker
            f.seek(frame_index)
            target_audio = f.read(self.chunk_length, dtype="float32")

        return (
            torch.tensor(input_audio).unsqueeze(0),
            torch.tensor(target_audio).unsqueeze(0),
        )

    def __len__(self) -> int:
        self.num_frames // self.chunk_length


class DistanceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mx20: MX20,
        chunk_length: int = 2048,
    ):
        super().__init__()
        self.mx20 = mx20
        self.chunk_length = chunk_length


    def setup(self, stage: str):
        dataset = RecordingDataset(mx20=self.mx20,chunk_length=self.chunk_length)
        training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [8,2])
        self.training_dataset = ConcatDataset(training_dataset)
        self.validation_dataset = ConcatDataset(validation_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    per_second = RecordingDataset(
        DAY_1_FOLDER,
        {"nt1": "67"},
        chunk_length=44100,
        prefix="day_1",
    )

    input_frames = 0
    for input_file in per_second.input_files:
        with sf.SoundFile(input_file, "r") as f:
            input_frames += f.frames
    print("Lossless length in seconds:", input_frames / 44100)

    print("Available length in seconds:", len(per_second))
    apparent_loss = per_second._target_loss / 44100
    print("Apparent loss in seconds", apparent_loss)

    actual_frames = 0
    for i in range(len(per_second)):
        input_audio, target_audio = per_second[i]
        actual_frames += len(input_audio)
        print(f"{i}/{len(per_second)}", end="\r")
    print("Actual length in seconds:", actual_frames)
    print("Actual length with loss:", actual_frames + apparent_loss)
