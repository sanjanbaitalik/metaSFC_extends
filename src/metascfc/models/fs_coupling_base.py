from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class FSCouplingModel(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, fc_graph, sc_graph) -> Dict[str, Any]:
        pass
