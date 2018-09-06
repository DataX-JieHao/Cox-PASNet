from DataLoader import load_data, load_pathway
from Train_for_Interpret import InterpretCoxPASNet
from Model import Cox_PASNet

import torch
import numpy as np


dtype = torch.FloatTensor
''' Net Settings'''
In_Nodes = 5567 ###number of genes
Pathway_Nodes = 860 ###number of pathways
Hidden_Nodes = 100 ###number of hidden nodes
Out_Nodes = 30 ###number of hidden nodes in the last hidden layer
''' Initialize '''
Initial_Learning_Rate = 0.03
L2_Lambda = 0.001
Num_EPOCHS = 15000
###sub-network setup
Dropout_Rate = [0.7, 0.5]

''' load data and pathway '''
pathway_mask = load_pathway("../data/pathway_mask.csv", dtype)
x, ytime, yevent, age = load_data("../data/entire_data.csv", dtype)

outpath = "../results/InterpretCoxPASNet.pt"
'''train Cox-PASNet for model interpretation'''
InterpretCoxPASNet(x, age, ytime, yevent, pathway_mask, \
					In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
					Initial_Learning_Rate, L2_Lambda, Num_EPOCHS, Dropout_Rate, outpath)

'''load trained Cox-PASNet'''
net = Cox_PASNet(In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, pathway_mask)
net.load_state_dict(torch.load(outpath))
###if gpu is being used
if torch.cuda.is_available():
	net.cuda()
###
'''save weights and node values into files individually'''
w_sc1 = net.sc1.weight.data.cpu().detach().numpy()
w_sc2 = net.sc2.weight.data.cpu().detach().numpy()
w_sc3 = net.sc3.weight.data.cpu().detach().numpy()
w_sc4 = net.sc4.weight.data.cpu().detach().numpy()
np.savetxt("../results/w_sc1.csv", w_sc1, delimiter = ",")
np.savetxt("../results/w_sc2.csv", w_sc2, delimiter = ",")
np.savetxt("../results/w_sc3.csv", w_sc3, delimiter = ",")
np.savetxt("../results/w_sc4.csv", w_sc4, delimiter = ",")

pathway_node = net.tanh(net.sc1(x))
hidden_node = net.tanh(net.sc2(pathway_node))
hidden_2_node = net.tanh(net.sc3(hidden_node))
x_cat = torch.cat((hidden_2_node, age), 1)
lin_pred = net.sc4(x_cat)

np.savetxt("../results/pathway_node.csv", pathway_node.cpu().detach().numpy(), delimiter = ",")
np.savetxt("../results/hidden_node.csv", hidden_node.cpu().detach().numpy(), delimiter = ",")
np.savetxt("../results/hidden_2_node.csv", x_cat.cpu().detach().numpy(), delimiter = ",")
np.savetxt("../results/lin_pred.csv", lin_pred.cpu().detach().numpy(), delimiter = ",")

