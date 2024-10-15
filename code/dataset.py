import os
import sys
import torch
from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.transforms import RandomNodeSplit, BaseTransform
from typing import Optional, Tuple, Union
from torch_geometric.data.storage import NodeStorage
from torch import Tensor
import graph

class RandomNodeSplit(BaseTransform):

    def __init__(
        self,
        split: str = "train_rest",
        num_splits: int = 1,
        num_train_per_class: int = 20,
        num_val: Union[int, float] = 0.2,
        num_test: Union[int, float] = 0.1,
        key: Optional[str] = "y",
    ):
        assert split in ['train_rest', 'test_rest', 'random']
        self.split = split
        self.num_splits = num_splits
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.key = key

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[self._split(store) for _ in range(self.num_splits)])

            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)

        return data

    def _split(self, store: NodeStorage) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = store.num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = round(num_nodes * self.num_val)
        else:
            num_val = self.num_val

        if isinstance(self.num_test, float):
            num_test = round(num_nodes * self.num_test)
        else:
            num_test = self.num_test

        if self.split == 'train_rest':
            perm = torch.randperm(num_nodes)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val:num_val + num_test]] = True
            train_mask[perm[num_val + num_test:]] = True
        else:
            y = getattr(store, self.key)
            num_classes = int(y.max().item()) + 1
            for c in range(num_classes):
                idx = (y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))]
                idx = idx[:self.num_train_per_class]
                train_mask[idx] = True

            remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            val_mask[remaining[:num_val]] = True

            if self.split == 'test_rest':
                test_mask[remaining[num_val:]] = True
            elif self.split == 'random':
                test_mask[remaining[num_val:num_val + num_test]] = True

        return train_mask, val_mask, test_mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(split={self.split})'

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
dataset = torch.load(proj_dir+'\\data\\graphs\\train_data_None_correlation_4neig.pth')  
transform = RandomNodeSplit(split='train_rest')
train_data = transform(dataset.get(0))
torch.save(train_data, proj_dir+'\\data\\dataset.pth')