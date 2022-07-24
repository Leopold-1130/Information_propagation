import factorgraph as fg
import numpy as np

class GraphConverter(object):
	@classmethod
	def planetoid_to_factor_graph(self, plaintoid_g, 
										num_classes: int, 
										edge_H: np.array,
										observed_nodes: [int],
										observed_node_class: [int],
								):
		edges 		= plaintoid_g.edge_index
		num_nodes 	= plaintoid_g.size()[0]
		factor_g 	= fg.Graph()

		for i in range(num_nodes):
			factor_g.rv(str(i), num_classes)

		for col in range(plaintoid_g.edge_index.shape[-1]):
			[node_A, node_B] = plaintoid_g.edge_index[:, col]
			node_A 	= node_A.item()
			node_B 	= node_B.item()
			factor_g.factor([str(node_A), str(node_B)], potential = edge_H)

		for each_observed_node, each_observed_node_class in zip(observed_nodes, observed_node_class):
			factor_g.factor([str(each_observed_node)], potential = np.array([1.0 if i == each_observed_node_class else 0.0 for i in range(num_classes)]))

		return factor_g