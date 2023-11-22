import torchaudio
import pytorch_lightning as pl
import soundfile as sf
import torch
import os
from torch.utils.data import ConcatDataset, Dataset, DataLoader



class RecordingDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        target_path: str,
        *,
        stride_length: int = 1024,
        chunk_length: int = 2048,
        half: bool = True,
    ):
        self.input_path = input_path
        self.target_path = target_path
        if not os.path.exists(self.input_path):
            raise Exception(f"File not found: {self.input_path}")
        if not os.path.exists(self.target_path):
            raise Exception(f"File not found: {self.target_path}")
        self.chunk_length = chunk_length
        self.stride_length = stride_length
        self.num_frames = torchaudio.info(self.input_path).num_frames
        self.half = half

    def __getitem__(self, marker: int):
        with sf.SoundFile(self.input_path, "r") as f:
            frame_index = self.stride_length * marker
            f.seek(frame_index)
            input_audio = f.read(self.chunk_length, dtype="float32", always_2d=True)
            input_audio = torch.tensor(input_audio.T)

        with sf.SoundFile(self.target_path, "r") as f:
            frame_index = self.stride_length * marker
            f.seek(frame_index)
            target_audio = f.read(self.chunk_length, dtype="float32", always_2d=True)
            target_audio = torch.tensor(target_audio.T)

        if self.half:
            input_audio = input_audio.half()
            target_audio = target_audio.half()

        return (
            input_audio,
            target_audio,
        )

    def __len__(self) -> int:
        return (self.num_frames - self.chunk_length) // self.stride_length


class MicroChangeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        chunk_length: int = 2048,
        stride_length: int = 1024,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = True,
        half: bool = True,
    ):
        super().__init__()
        self.chunk_length = chunk_length
        self.stride_length = stride_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.half = half

    def setup(self, stage: str):
        datasets = sum([
            [
                RecordingDataset(
                    input_path=f"../data/{day}/{x}",
                    target_path=f"../data/{day}/67_near.wav",
                    chunk_length=self.chunk_length,
                    stride_length=self.stride_length,
                    half=self.half,
                )
                for x in [
                    "103_far.wav",
                    "103_middle.wav",
                    "103_near.wav",
                    "269_far.wav",
                    "4040_middle.wav",
                    "414_far.wav",
                    "414_near.wav",
                    "87_far.wav",
                    "87_near.wav",
                    "nt1_middle.wav",
                ]
            ]
            for day in ["day1"]
        ],[])
        dataset = ConcatDataset(datasets)
        training_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [0.8, 0.2]
        )
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

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
