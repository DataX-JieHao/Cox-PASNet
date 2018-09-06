import torch
import torch.nn as nn

class Cox_PASNet(nn.Module):
	def __init__(self, In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, Pathway_Mask):
		super(Cox_PASNet, self).__init__()
		self.tanh = nn.Tanh()
		self.pathway_mask = Pathway_Mask
		###gene layer --> pathway layer
		self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes)
		###pathway layer --> hidden layer
		self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
		###hidden layer --> hidden layer 2
		self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes, bias=False)
		###hidden layer 2 + age --> Cox layer
		self.sc4 = nn.Linear(Out_Nodes+1, 1, bias = False)
		self.sc4.weight.data.uniform_(-0.001, 0.001)
		###randomly select a small sub-network
		self.do_m1 = torch.ones(Pathway_Nodes)
		self.do_m2 = torch.ones(Hidden_Nodes)
		###if gpu is being used
		if torch.cuda.is_available():
			self.do_m1 = self.do_m1.cuda()
			self.do_m2 = self.do_m2.cuda()
		###

	def forward(self, x_1, x_2):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
		x_1 = self.tanh(self.sc1(x_1))
		if self.training == True: ###construct a small sub-network for training only
			x_1 = x_1.mul(self.do_m1)
		x_1 = self.tanh(self.sc2(x_1))
		if self.training == True: ###construct a small sub-network for training only
			x_1 = x_1.mul(self.do_m2)
		x_1 = self.tanh(self.sc3(x_1))
		###combine age with hidden layer 2
		x_cat = torch.cat((x_1, x_2), 1)
		lin_pred = self.sc4(x_cat)
		
		return lin_pred

