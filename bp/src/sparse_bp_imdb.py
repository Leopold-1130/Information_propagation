import dgl
import os
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import IMDB
from torch_geometric.utils import to_scipy_sparse_matrix

if __name__ == "__main__":
	geometric_graph 		= IMDB(root='/tmp/imdb')[0]
	
	import IPython
	IPython.embed()