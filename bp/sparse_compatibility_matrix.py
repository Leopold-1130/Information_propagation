"""
We implement the strategy employed by "Factorized Graph Representations forSemi-Supervised Learning from Sparse Data" (Kumar et al.)
in order to obtain the compatibility matrix for belief propagation algorithms
"""

import dgl
import networkx as nx
import torch
import torch.nn as nn
import os

if __name__ == "__main__":
	G 				= nx.read_gpickle("/Users/johan.kok/Downloads/year_2022_month_04_day_21_hour_07_SIN_4W.gpickle")
	dgl_graph 		= dgl.from_networkx(G, node_attrs = ["v_class"])

	adj_mat 		= dgl_graph.adjacency_matrix()
	feats 			= dgl_graph.ndata["v_class"]
	num_classes 	= int(feats.max().item()) +1
	feats_1_hot 	= nn.functional.one_hot(feats, num_classes = num_classes).type(torch.FloatTensor)

	adj_mat_feats 	= torch.matmul(adj_mat, feats_1_hot)
	H 				= torch.rand((num_classes, num_classes), requires_grad = True)

	loss_fnct 		= nn.MSELoss(reduction = "mean")

	optimizer       = torch.optim.Adam( H.parameters(), lr = 2e-7, betas = (0.9, 0.999), 
                                        eps = 1e-08, weight_decay = 0, amsgrad = False)

	for _ in range(100):
		loss = loss_fnct(feats_1_hot, torch.matmul(adj_mat_feats, H))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(loss)