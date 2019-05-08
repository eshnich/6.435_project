import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import time
import math
import matplotlib.pyplot as plt
import numpy as np
from load_sprites import sprites_act

from sequential_cnn_vae import CNNSequentialVAE
from classifier import CNN
from action_classifier import CNN as actionCNN

if __name__ == '__main__':

	latent_dimz = 32 # total of 9 different actions 
	latent_dimf = 256 # total number of types of characters
	net = CNNSequentialVAE(3, 64, 64, 8, latent_dimz, latent_dimf)
	net.load_state_dict(torch.load('SavedModels/post_train_test.pt', map_location='cpu'))
	
	X_train, X_test, A_train, A_test, D_train, D_test = sprites_act('', return_labels=True)
	X_test = torch.from_numpy(X_test)
	X_test = X_test.transpose(3,4).transpose(2,3)

	test_samples = 100
	data = X_test.narrow(0, 0, test_samples)

	classify_attributes = False
	
	if classify_attributes:
		attr = 3
		classifier = CNN(3, 64,64, 8)
		classifier.load_state_dict(torch.load('SavedModels/classifier_attr_%d.pt' % attr, map_location='cpu'))
		test_labels = [[i for i in range(6) if A_test[j][0][attr][i] ==1][0] for j in range(len(A_test))]
		test_labels = test_labels[:test_samples]

		f_params = net.encode_f(data.transpose(0,1))
		f = net.reparameterize(f_params[0], f_params[1])
		z = torch.randn(8, test_samples, latent_dimz)

	else:
		classifier = actionCNN(3, 64, 64, 8)#update this
		classifier.load_state_dict(torch.load('SavedModels/classifier_action.pt', map_location='cpu'))
		test_labels = [[i for i in range(9) if D_test[j][0][i] ==1][0] for j in range(len(D_test))]
		test_labels = test_labels[:test_samples]
		z = []
		for i in range(8):
			x = data.transpose(0,1)
			mu_z_i, logvar_z_i = net.encode_z(x[i])
			z.append(net.reparameterize(mu_z_i,logvar_z_i))

		f = torch.randn(test_samples, latent_dimf)

	recon = torch.cat([net.decode(z[i], f) for i in range(len(z))], 0).view(8, -1, 3, 64, 64).transpose(0,1)
	#want to compare recon_x vs data
	pred_data = [classifier(data[i].view(1,8,3,64,64).transpose(0,1)).max(1)[1].item() for i in range(test_samples)]
	pred_recon = [classifier(recon[i].view(1,8,3,64,64).transpose(0,1)).max(1)[1].item() for i in range(test_samples)]
	print("agreement = ", sum([pred_data[i] == pred_recon[i] for i in range(test_samples)])/float(test_samples))
