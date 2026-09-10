from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class EntryModule(ABC):
    name: str = "base"

    @abstractmethod
    def check(self, df: pd.DataFrame, ctx, config) -> Optional[dict]:
        raise NotImplementedError
