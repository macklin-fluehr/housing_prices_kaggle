import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from typing import List

from typing import Union


class RegressorDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        labels: Union[pd.Series, None],
        categorical_col_names: List[str],
    ) -> None:

        self.data = data.reset_index(drop=True)

        if labels is not None:
            self.labels = labels.reset_index(drop=True)
        else:
            self.labels = None

        self.categorical_col_names = categorical_col_names

        self.cont = (
            self.data.drop(categorical_col_names, axis=1)
            .copy()
            .astype(np.float32)
        )
        self.cat = self.data[categorical_col_names].copy().astype(np.int64)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx):

        # Grabbing Continuous and Categorical Data for Each row
        cont = self.cont.loc[idx].values
        cat = self.cat.loc[idx].values

        if self.labels is None:
            return (cont, cat)
        else:
            return (cont, cat), self.labels.loc[idx]
