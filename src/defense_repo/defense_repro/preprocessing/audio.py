import torch


def fft_features(waveform: torch.Tensor, n_fft: int | None = None) -> torch.Tensor:
    """
    Magnitude FFT features, corresponding to the slide's '音频文件：傅里叶变换'.
    waveform: [..., T]
    """
    if n_fft is None:
        n_fft = waveform.shape[-1]
    spectrum = torch.fft.rfft(waveform.float(), n=n_fft, dim=-1)
    return torch.log1p(torch.abs(spectrum))
