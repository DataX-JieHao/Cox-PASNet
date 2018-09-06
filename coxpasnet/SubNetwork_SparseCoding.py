import torch 
import numpy as np
import math

def dropout_mask(n_node, drop_p):
	'''Construct a binary matrix to randomly drop nodes in a layer.
	Input:
		n_node: number of nodes in the layer.
		drop_p: the probability that a node is to be dropped.
	Output:
		mask: a binary matrix, where 1 --> keep the node; 0 --> drop the node.
	'''
	keep_p = 1.0 - drop_p
	mask = torch.Tensor(np.random.binomial(1, keep_p, size=n_node))
	###if gpu is being used
	if torch.cuda.is_available():
		mask = mask.cuda()
	###
	return mask

def s_mask(sparse_level, param_matrix, nonzero_param_1D, dtype):
	'''Construct a binary matrix w.r.t. a sparsity level of weights between two consecutive layers
	Input:
		sparse_level: a percentage value in [0, 100) represents the proportion of weights in a sub-network to be dropped.
		param_matrix: a weight matrix for entrie network.
		nonzero_param_1D: 1D of non-zero 'param_matrix' (which is the weights selected from a sub-network).
		dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor).
	Output:
		param_mask: a binary matrix, where 1 --> keep the node; 0 --> drop the node.
	'''
	###take the absolute values of param_1D 
	non_neg_param_1D = torch.abs(nonzero_param_1D)
	###obtain the number of params
	num_param = nonzero_param_1D.size(0)
	###obtain the kth number based on sparse_level
	top_k = math.ceil(num_param*(100-sparse_level)*0.01)
	###obtain the k largest params
	sorted_non_neg_param_1D, indices = torch.topk(non_neg_param_1D, top_k)
	param_mask = torch.abs(param_matrix) > sorted_non_neg_param_1D.min()
	param_mask = param_mask.type(dtype)
	###if gpu is being used
	if torch.cuda.is_available():
		param_mask = param_mask.cuda()
	###
	return param_mask


