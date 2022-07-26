"""
Implementation based on the paper: "Linearized and Single-Pass Belief Propagation" by Gatterbauer et al.
"""

import dgl
import os
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def build_explicit_belief():
	pass

if __name__ == "__main__":
	geometric_graph 		= Planetoid(root='/tmp/pubmed', name='Pubmed')[0]
	edge_H 					= torch.load(os.path.join(os.getcwd(), "matrices", "pubmed", "H.pt"))

	adj_mat 				= to_scipy_sparse_matrix(geometric_graph.edge_index)
	adj_mat 				= dgl.from_scipy(adj_mat).adjacency_matrix()
	node_degrees			= torch.sparse.sum(adj_mat, dim = 1).values()
	degree_mat 				= torch.diag(node_degrees)

	num_classes 			= geometric_graph.y.max() +1
	observed_belief 		= geometric_graph.y * geometric_graph.train_mask
	observed_belief_1_hot 	= nn.functional.one_hot(observed_belief, num_classes = num_classes).type(torch.FloatTensor)

	posterior_belief 		= torch.zeros(observed_belief_1_hot.shape)
	loss_fnct 				= nn.MSELoss(reduction = "mean")
	
	for _ in range(100):
		prior_belief 			= posterior_belief

		adj_belief 				= torch.matmul(adj_mat, posterior_belief)
		adj_belief_H 			= torch.matmul(adj_belief, edge_H)

		degree_belief 			= torch.matmul(degree_mat, posterior_belief)
		degree_belief_H_sq 		= torch.matmul(degree_belief, torch.square(edge_H))

		posterior_belief 		= observed_belief_1_hot + adj_belief_H - degree_belief_H_sq

		loss = loss_fnct(posterior_belief, prior_belief)
		print(loss)