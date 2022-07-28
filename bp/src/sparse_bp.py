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
	edge_H_norm 			= torch.norm(edge_H)

	adj_mat 				= to_scipy_sparse_matrix(geometric_graph.edge_index)
	adj_mat 				= dgl.from_scipy(adj_mat).adjacency_matrix()
	adj_mat_norm 			= torch.norm(adj_mat)

	edge_H_scaled 			= edge_H * 0.20

	node_degrees			= torch.sparse.sum(adj_mat, dim = 1).values()
	degree_mat 				= torch.diag(node_degrees)

	num_classes 			= geometric_graph.y.max() +1
	observed_belief 		= geometric_graph.y * (geometric_graph.val_mask == False)
	observed_belief_1_hot 	= nn.functional.one_hot(observed_belief, num_classes = num_classes).type(torch.FloatTensor)

	posterior_belief 		= torch.zeros(observed_belief_1_hot.shape)
	loss_fnct 				= nn.MSELoss(reduction = "mean")
	
	for _ in range(50):
		prior_belief 			= posterior_belief

		adj_belief 				= torch.matmul(adj_mat, posterior_belief)
		adj_belief_H 			= torch.matmul(adj_belief, edge_H_scaled)

		degree_belief 			= torch.matmul(degree_mat, posterior_belief)
		degree_belief_H_sq 		= torch.matmul(degree_belief, torch.square(edge_H_scaled))

		posterior_belief 		= observed_belief_1_hot + adj_belief_H - degree_belief_H_sq
		posterior_belief[(geometric_graph.val_mask == False)] = observed_belief_1_hot[(geometric_graph.val_mask == False)]

		loss = loss_fnct(posterior_belief, prior_belief)
		print(loss)

	cross_entrooy_loss_fnct = nn.CrossEntropyLoss(reduction = "mean")
	ground_truth 		= geometric_graph.y[geometric_graph.val_mask == True]
	predicted_labels 	= posterior_belief[geometric_graph.val_mask == True]
	cross_entrooy_loss 	= cross_entrooy_loss_fnct(predicted_labels, ground_truth)
	print(f"Cross Entropy loss {cross_entrooy_loss}")