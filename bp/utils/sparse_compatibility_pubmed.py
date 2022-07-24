"""
We implement the strategy employed by "Factorized Graph Representations forSemi-Supervised Learning from Sparse Data" (Kumar et al.)
in order to obtain the compatibility matrix for belief propagation algorithms
"""

import dgl
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

if __name__ == "__main__":
	LEARNING_RATE 	= 1e-5
	EARLY_TERMINATE = 3

	dataset 		= Planetoid(root='/tmp/pubmed', name='Pubmed')[0]
	num_classes 	= dataset.y.max() +1
	feats 			= dataset.y
	feats_1_hot 	= nn.functional.one_hot(feats, num_classes = num_classes).type(torch.FloatTensor)
	adj_mat 		= to_scipy_sparse_matrix(dataset.edge_index)
	adj_mat 		= dgl.from_scipy(adj_mat).adjacency_matrix()

	adj_mat_feats 	= torch.matmul(adj_mat, feats_1_hot)
	H 				= torch.rand((num_classes, num_classes), requires_grad = True)

	loss_fnct 		= nn.MSELoss(reduction = "mean")
	min_loss 		= sys.maxsize
	min_H 			= H
	loss_increasing_count = 0

	for _ in range(1000):
		loss = loss_fnct(feats_1_hot, torch.matmul(adj_mat_feats, H))
		loss.backward()
		print(loss)

		if loss < min_loss:
			min_loss 	= loss 
			min_H 		= H
			loss_increasing_count = 0

		else:
			loss_increasing_count += 1

		with torch.no_grad():
			H 	-= LEARNING_RATE * H.grad

		if loss_increasing_count >= EARLY_TERMINATE:
			break

	print(min_loss, min_H)
	torch.save(min_H, os.path.join(os.getcwd(), "matrices", "pubmed", "H.pt"))