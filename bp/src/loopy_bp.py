import numpy as np
import os
import torch
from torch_geometric.datasets import Planetoid
from utils.GraphConverter import GraphConverter

if __name__ == "__main__":
	plaintoid_g = Planetoid(root='/tmp/pubmed', name='Pubmed')[0]
	edge_H 		= torch.load(os.path.join(os.getcwd(), "matrices", "pubmed", "H.pt"))
	edge_H 		= edge_H.detach().numpy()

	factor_g 	= GraphConverter.planetoid_to_factor_graph(	plaintoid_g 		= plaintoid_g, 
															num_classes 		= plaintoid_g.y.max() +1,
															edge_H 				= edge_H,
															observed_nodes 		= [14, 2],
															observed_node_class = [0, 1]
														)

	iters, converged = factor_g.lbp(normalize = True, max_iters = 100)
	print(f"Iters: {iters}, Converged {converged}")