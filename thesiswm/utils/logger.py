from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, writer: SummaryWriter):
        self.w = writer

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.w.add_scalar(tag, float(value), int(step))
