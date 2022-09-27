import networkx as nx
import numpy as np
import os
import random
import torch
import torch.nn as nn
from utils.GraphConverter import GraphConverter

if __name__ == "__main__":
	nx_G 					= nx.read_gpickle("../datasets/homogenous_dynamic/elliptic_49.gpickle")
	edge_H 					= torch.load(os.path.join(os.getcwd(), "matrices", "elliptic", "H_49.pt"))
	edge_H 					= edge_H.detach().numpy()

	all_nodes 				= set(nx_G.nodes())
	all_nodes_class 		= [nx_G.nodes()[each_node]["cls"] for each_node in all_nodes]
	max_classes 			= max(all_nodes_class) +1

	observed_nodes 			= []
	observed_nodes_class 	= []

	unobserved_nodes 		= []
	unobserved_nodes_class 	= []

	for each_class in range(max_classes):
		nodes_at_class 			= [x for x,y in nx_G.nodes(data = True) if y['cls'] == each_class]
		
		_observed_nodes 		= random.sample(nodes_at_class, int(0.9 * len(nodes_at_class)))
		_unobserved_nodes 		= list(set(nodes_at_class) - set(_observed_nodes))

		observed_nodes 			+= _observed_nodes
		unobserved_nodes 		+= _unobserved_nodes

		observed_nodes_class 	+= [each_class for _ in range(len(_observed_nodes))]
		unobserved_nodes_class 	+= [each_class for _ in range(len(_unobserved_nodes))]

	factor_g 	= GraphConverter.networkx_to_factor_graph(	nx_G 				= nx_G, 
															num_classes 		= max_classes,
															edge_H 				= edge_H,
															observed_nodes 		= observed_nodes,
															observed_node_class = unobserved_nodes_class
														)

	iters, converged = factor_g.lbp(normalize = False, max_iters = 100)
	print(f"Iters: {iters}, Converged {converged}")

	marginals 				= factor_g.rv_marginals()
	predicted_class 		= filter(lambda x: int(x[0].name) in unobserved_nodes, marginals)
	predicted_class 		= list(map(lambda x: x[-1], predicted_class))
	predicted_class 		= torch.tensor(predicted_class)
	unobserved_nodes_class  = torch.tensor(unobserved_nodes_class)

	loss_fnct 				= nn.CrossEntropyLoss(reduction = "mean")
	loss 					= loss_fnct(predicted_class, unobserved_nodes_class)
	print(loss)

	(_, best_predicted_class) 	= torch.max(predicted_class, dim = 1)
	accuracy 					= torch.sum(best_predicted_class == unobserved_nodes_class) / len(unobserved_nodes_class)
	print(f"Accuracy {accuracy}")