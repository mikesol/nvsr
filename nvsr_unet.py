import sys

sys.path.append("/vol/research/dcase2022/sr_eval_vctk/testees")

from auraloss.freq import MultiResolutionSTFTLoss
import torch.utils
import torch.nn as nn
import torch.utils.data
from voicefixer import Vocoder
import os
import pytorch_lightning as pl
from fDomainHelper import FDomainHelper
from mel_scale import MelScale
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

EPS = 1e-9

def to_log(input):
    assert torch.sum(input < 0) == 0, (
        str(input) + " has negative values counts " + str(torch.sum(input < 0))
    )
    return torch.log10(torch.clip(input, min=1e-8))

def from_log(input):
    input = torch.clip(input, min=-np.inf, max=5)
    return 10**input

def trim_center(est, ref):
    diff = np.abs(est.shape[-1] - ref.shape[-1])
    if est.shape[-1] == ref.shape[-1]:
        return est, ref
    elif est.shape[-1] > ref.shape[-1]:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., int(diff // 2) : -int(diff // 2)], ref
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref
    else:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est, ref[..., int(diff // 2) : -int(diff // 2)]
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class BN_GRU(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer=1,
        bidirectional=False,
        batchnorm=True,
        dropout=0.0,
    ):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

    def forward(self, inputs):
        # (batch, 1, seq, feature)
        if self.batchnorm:
            inputs = self.bn(inputs)
        out, _ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class NVSR(pl.LightningModule):
    def __init__(self, channels):
        super(NVSR, self).__init__()

        model_name = "unet"

        self.channels = channels

        self.vocoder = Vocoder(sample_rate=44100)

        self.downsample_ratio = 2**6  # This number equals 2^{#encoder_blcoks}

        self.loss = MultiResolutionSTFTLoss()

        self.f_helper = FDomainHelper(
            window_size=2048,
            hop_size=441,
            center=True,
            pad_mode="reflect",
            window="hann",
            freeze_parameters=True,
        )

        self.mel = MelScale(n_mels=128, sample_rate=44100, n_stft=2048 // 2 + 1)

        # masking
        self.generator = Generator(model_name)
        # print(get_n_params(self.vocoder))
        # print(get_n_params(self.generator))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        _, mel = self.pre(x)
        out = self(mel)
        mel2 = from_log(out["mel"])
        out = self.vocoder(mel2)
        out, _ = trim_center(out, x)
        # for l in self.loss.stft_losses:
        #     l.window = l.window.to("cuda:0")
        print(out.device, y.device)
        loss = self.loss(out, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        _, mel = self.pre(x)
        out = self(mel)
        mel2 = from_log(out["mel"])
        out = self.vocoder(mel2)
        out, _ = trim_center(out, x)
        # for l in self.loss.stft_losses:
        #     l.window = l.window.to("cuda:0")
        print(out.device, y.device)
        loss = self.loss(out, y)
        self.log("training_loss", loss)
        return loss

    def get_vocoder(self):
        return self.vocoder

    def get_f_helper(self):
        return self.f_helper

    def pre(self, input):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return sp, mel_orig

    def forward(self, mel_orig):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        return self.generator(mel_orig)


def to_log(input):
    assert torch.sum(input < 0) == 0, (
        str(input) + " has negative values counts " + str(torch.sum(input < 0))
    )
    return torch.log10(torch.clip(input, min=1e-8))


def from_log(input):
    input = torch.clip(input, min=-np.inf, max=5)
    return 10**input


class BN_GRU(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer=1,
        bidirectional=False,
        batchnorm=True,
        dropout=0.0,
    ):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

    def forward(self, inputs):
        # (batch, 1, seq, feature)
        if self.batchnorm:
            inputs = self.bn(inputs)
        out, _ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)


class Generator(nn.Module):
    def __init__(self, model_name="unet"):
        super(Generator, self).__init__()
        if model_name == "unet":
            from components.unet import UNetResComplex_100Mb

            self.analysis_module = UNetResComplex_100Mb(channels=1)
        elif model_name == "unet_small":
            from components.unet_small import UNetResComplex_100Mb

            self.analysis_module = UNetResComplex_100Mb(channels=1)
        elif model_name == "bigru":
            n_mel = 128
            self.analysis_module = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Linear(n_mel, n_mel * 2),
                BN_GRU(
                    input_dim=n_mel * 2,
                    hidden_dim=n_mel * 2,
                    bidirectional=True,
                    layer=2,
                ),
                nn.ReLU(),
                nn.Linear(n_mel * 4, n_mel * 2),
                nn.ReLU(),
                nn.Linear(n_mel * 2, n_mel),
            )
        elif model_name == "dnn":
            n_mel = 128
            self.analysis_module = nn.Sequential(
                nn.Linear(n_mel, n_mel * 2),
                nn.ReLU(),
                nn.BatchNorm2d(1),
                nn.Linear(n_mel * 2, n_mel * 4),
                nn.ReLU(),
                nn.BatchNorm2d(1),
                nn.Linear(n_mel * 4, n_mel * 4),
                nn.ReLU(),
                nn.BatchNorm2d(1),
                nn.Linear(n_mel * 4, n_mel * 2),
                nn.ReLU(),
                nn.Linear(n_mel * 2, n_mel),
            )
        else:
            pass  # todo warning

    def forward(self, mel_orig):
        out = self.analysis_module(to_log(mel_orig))
        if type(out) == type({}):
            out = out["mel"]
        mel = out + to_log(mel_orig)
        return {"mel": mel}
