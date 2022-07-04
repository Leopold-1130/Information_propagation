# Datasets
[OGB-products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products)  
[OGB-proteins](https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins)   
[OGB-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)     
[OGB-papers](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M)   
[Cora/CiteSeer/Pubmed](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid)   

# How to use
replace d_name with  the dataset name (e.g., "ogbn-proteins").

```
from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name = d_name) 

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0] # pyg graph object
```
