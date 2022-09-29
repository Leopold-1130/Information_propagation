"""
Implementation based on the paper: "Linearized and Single-Pass Belief Propagation" by Gatterbauer et al.
"""

import dgl
import networkx as nx
import os
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

if __name__ == "__main__":
	nx_G 							= nx.read_gpickle("../datasets/homogenous_dynamic/elliptic_49.gpickle")
	dgl_G 							= dgl.from_networkx(nx_G, node_attrs = ["cls"])

	edge_H 							= torch.load(os.path.join(os.getcwd(), "matrices", "elliptic", "H_49.pt"))
	adj_mat 						= dgl_G.adjacency_matrix()

	node_degrees					= torch.sparse.sum(adj_mat, dim = 1).values()
	degree_mat 						= torch.diag(node_degrees)

	all_nodes_class 				= dgl_G.ndata["cls"].long()
	validation_count 				= int(0.2 * len(all_nodes_class))
	max_classes 					= max(all_nodes_class) +1

	training_mask 					= torch.zeros(len(all_nodes_class))

	for each_class in range(max_classes):
		indx_of_cls 					= torch.nonzero(all_nodes_class == each_class).flatten()
		training_smpls 					= indx_of_cls[ : len(indx_of_cls) - validation_count // max_classes]
		training_mask[training_smpls] 	= 1

	val_mask 						= 1 - training_mask
	observed_belief 				= all_nodes_class * training_mask
	observed_belief_1_hot 			= nn.functional.one_hot(observed_belief.long(), num_classes = max_classes).type(torch.FloatTensor)
	observed_belief_1_hot_centered 	= observed_belief_1_hot - 1 / max_classes

	edge_H_centered 				= edge_H - 1 / max_classes
	posterior_belief 				= torch.zeros(observed_belief_1_hot.shape)
	loss_fnct 						= nn.MSELoss(reduction = "mean")

	for _ in range(20):
		prior_belief 			= posterior_belief
		adj_belief 				= torch.matmul(adj_mat, posterior_belief)
		adj_belief_H 			= torch.matmul(adj_belief, edge_H_centered)

		degree_belief 			= torch.matmul(degree_mat, posterior_belief)
		degree_belief_H_sq 		= torch.matmul(degree_belief, torch.square(edge_H_centered))

		posterior_belief 		= observed_belief_1_hot_centered + adj_belief_H - degree_belief_H_sq
		posterior_belief[training_mask == 1] = observed_belief_1_hot_centered[training_mask == 1]

		loss = loss_fnct(posterior_belief[val_mask == 1], prior_belief[val_mask == 1])
		print(loss)

	cross_entropy_loss_fnct = nn.CrossEntropyLoss(reduction = "mean")
	ground_truth 		= all_nodes_class[val_mask == True]
	predicted_labels 	= posterior_belief[val_mask == True]
	cross_entropy_loss 	= cross_entropy_loss_fnct(predicted_labels, ground_truth)
	print(f"Cross Entropy loss {cross_entropy_loss}")

	(_, best_predicted_class) = torch.max(predicted_labels, dim = 1)
	accuracy 			= torch.sum(best_predicted_class == ground_truth) / len(ground_truth)
	print(f"Accuracy {accuracy}")
