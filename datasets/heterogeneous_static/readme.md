# Datasets
The graph, prediction task and the dataset splitting way can be seen in the links below.  
[ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)   
[IMDB](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.IMDB)     
[DBLP](https://paperswithcode.com/dataset/dblp)   
[LastFM](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.LastFM)

# How to use
## OGB Datasets
Replace d_name with  the dataset name (e.g., "ogbn-proteins").  
The sample code is shown in this link https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py.

```
from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name = d_name) 

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0] # pyg graph object
```

## Pytorch-geometric Dataset

```
from torch_geometric.datasets import IMDB, LastFM

imdb = IMDB(root='./IMBD') 
lastfm = LastFM(root='./LastFM')

```

