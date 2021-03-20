import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple


class EmbeddingBaseModel(nn.Module):
    def __init__(
        self,
        n_cont: int,
        embedding_sizes: List[Tuple[int, int]],
        out_dim: int,
    ):
        """

        Inputs:
            n_cont (int) - the number of continuous features to process
            embedding_sizes (List[Tuple[int, int]]) - this contains information
                            on all the categorical features. The first number
                            in the tuple tells us how many categories are
                            within that feature. The second number describes
                            how large the embedding for that feature will be.
            out_dim - the size of the embedding upon exit
        """
        super(EmbeddingBaseModel, self).__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(categories, size)
                for categories, size in embedding_sizes
            ]
        )
        n_emb: int = sum(
            e.embedding_dim for e in self.embeddings
        )  # length of all embeddings combined

        self.n_emb, self.n_cont = n_emb, n_cont

        layer_1_units = 16
        layer_2_units = 16
        layer_3_units = 16

        self.lin1 = nn.Linear(self.n_emb + self.n_cont, layer_1_units)
        self.lin2 = nn.Linear(layer_1_units, layer_2_units)
        self.lin3 = nn.Linear(layer_2_units, layer_3_units)
        self.lin4 = nn.Linear(layer_3_units, out_dim)

        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(layer_1_units)
        self.bn3 = nn.BatchNorm1d(layer_2_units)
        self.bn4 = nn.BatchNorm1d(layer_3_units)

        self.emb_drop = nn.Dropout(0.1)

        self.drops = nn.Dropout(0.2)

    def forward(self, x_cont, x_cat):
        """
        Returns an embedding of our input features.

        Inputs:
            x_cont -
            x_cat -
        """

        # Cats
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        # x = self.emb_drop(x)

        # Cont
        x2 = self.bn1(x_cont)

        # Concat
        x = torch.cat([x, x2], 1)

        # Linear Layer 1
        x = F.relu(self.lin1(x))
        # x = self.drops(x)
        # x = self.bn2(x)

        # Linear Layer 2
        x = F.relu(self.lin2(x))
        # x = self.drops(x)
        # x = self.bn3(x)

        # Linear Layer 3
        x = F.relu(self.lin3(x))
        # x = self.drops(x)
        # x = self.bn4(x)

        x = self.lin4(x)

        return x
