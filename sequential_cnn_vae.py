import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from IPython import embed
from load_sprites import sprites_act
import time
import math
import matplotlib.pyplot as plt


if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")
        device = torch.device("cuda:0")
        print("DEVICE:", device)
else:
        print("CUDE IS NOT AVAILABLE")
        device = torch.device("cpu")
        print(device)


# probably want to figure out how to use a more sophisticated RNN for higher dimensional data (LSTM, etc.)
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		return output, hidden

	def initHidden(self, batch_size):
		return torch.zeros(batch_size, self.hidden_size)

class CNNSequentialVAE(nn.Module):
	def __init__(self, image_channels, x_height, x_width, no_of_frames, latent_dimz, latent_dimf):
		super(CNNSequentialVAE, self).__init__()
		
		self.image_channels = image_channels # channels - 3 for RBG images
		self.x_height = x_height # height of input image
		self.x_width = x_width # width of the input image
		self.frames = no_of_frames # number of frames - not sure how to use batch
		self.latent_dimz = latent_dimz # dimension of z
		self.latent_dimf = latent_dimf # dimension of f
		
		# for the encoder
		self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding = (1,1))
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding = (1,1))
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=2)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=2)	
		
		#for the encoder for z, q(z_t|x_t)
		encoder_hidden_dim = 512
		self.hidden_layer = nn.Linear(in_features = 256*2*2, out_features = encoder_hidden_dim)
		self.fc11 = nn.Linear(in_features=encoder_hidden_dim, out_features=self.latent_dimz)
		self.fc12 = nn.Linear(in_features=encoder_hidden_dim, out_features=self.latent_dimz)

		#for the encoder for f, q(f|x_1:T)
		self.fc21 = nn.Linear(in_features=256, out_features=self.latent_dimf)
		self.fc22 = nn.Linear(in_features=256, out_features=self.latent_dimf)
		
		RNN_hidden_dim = 512
		self.RNN = RNN(256*2*2, RNN_hidden_dim, 2*self.latent_dimf)#.cuda() 
			
		# for the decoder
		decoder_mlp_hidden_dim = 512
		self.fc0 = nn.Linear(in_features=self.latent_dimz+self.latent_dimf, out_features = decoder_mlp_hidden_dim)
		self.fc1 = nn.Linear(in_features=decoder_mlp_hidden_dim, out_features=256*2*2)
		self.conv_t1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)

		self.conv_t2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2)

		self.conv_t3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding = (1,1))

		self.conv_t4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=1, padding = (1,1))

		self.sigmoid = nn.Sigmoid()

		# for the prior p(z_t|z_<t)
		RNN_hidden_dim2 = 3
		self.RNN_prior = RNN(self.latent_dimz, RNN_hidden_dim2, 2*self.latent_dimz)

	#encode visual features of x
	def encode_visual_features(self, x):
		vis = []

		for i in range(self.frames):
			h = x[i].view(-1, self.image_channels, self.x_height, self.x_width)
			h = F.elu(self.conv1(h))
			h = F.max_pool2d(F.elu(self.conv2(h)), kernel_size = (2,2))
			h = F.max_pool2d(F.elu(self.conv3(h)), kernel_size = (2,2))
			h = F.max_pool2d(F.elu(self.conv4(h)), kernel_size = (2,2))
			h = h.view(-1, 256*2*2)#.cuda()
			vis.append(h)
		return vis

	# compute parameters for q(z_t | x_t)
	def encode_z(self, x):
		h = F.elu(self.hidden_layer(x))
		mu = self.fc11(h)
		logvar = self.fc12(h)
		return mu, logvar
				
	
	# compute parameters for q(f | x[1:T])
	def encode_f(self, x):
		hidden = self.RNN.initHidden(x[0].size()[0])#.cuda()
		for i in range(self.frames):
			h = x[i]
			output, hidden = self.RNN(h, hidden)
		mu, logvar = torch.split(output, self.latent_dimf, dim=1)
		return mu, logvar

	def prior(self, z):
		hidden = self.RNN_prior.initHidden(z[0].size()[0])#.cuda()
		mu_prior = []
		logvar_prior = []
		for i in range(len(z)):
			output, hidden = self.RNN_prior(z[i], hidden)
			mu, logvar = torch.split(output, self.latent_dimz, dim=1)
			mu_prior.append(mu)
			logvar_prior.append(logvar)
		return mu_prior, logvar_prior

	# reparametrization trick for sampling from a Gaussian
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

    # compute parameters for p(x_t | z_t,f)
	def decode(self, z, f):
		combined = torch.cat((z,f), 1)
		combined = combined.view(-1, self.latent_dimz+self.latent_dimf)
		x = F.elu(self.fc0(combined))
		x = self.fc1(x)
		x = x.view(-1, 256, 2, 2)
		#embed()
		x = F.interpolate(F.elu(self.conv_t1(x)), scale_factor=2, mode = 'bilinear', align_corners = False)
		x = F.interpolate(F.elu(self.conv_t2(x)), scale_factor=2, mode = 'bilinear', align_corners = False)
		x = F.interpolate(F.elu(self.conv_t3(x)), scale_factor=2, mode = 'bilinear', align_corners = False)
		x = self.conv_t4(x)
		x = self.sigmoid(x)
		return x

	# forward pass of the algorithm 
    # returns:
    #       - recon_x: the outputted value of the VAE
    #       - mu_f, logvar_f: parameters for q(f | x_1:T), a Gaussian
    #       - mu_prior, logvar_prior: parameters for p(z_t|z_<t), a Gaussian
    #       - mu_z, logvar_z: parameters for q(z_t | x_t), a Gaussian
	def forward(self, x):

		vis = self.encode_visual_features(x)
		mu_f, logvar_f = self.encode_f(vis)
		f = self.reparameterize(mu_f, logvar_f)
		z = []
		mu_z = []
		logvar_z = []
		for i in range(self.frames):
			mu_z_i, logvar_z_i = self.encode_z(vis[i])
			z.append(self.reparameterize(mu_z_i,logvar_z_i))
			mu_z.append(mu_z_i)
			logvar_z.append(logvar_z_i)

		#recon_x = [self.decode(z[i].cuda(), f.cuda()) for i in range(len(z))]
		recon_x = [self.decode(z[i], f) for i in range(len(z))]

		#mu_prior, logvar_prior = self.prior(z)
		return recon_x, mu_f, logvar_f, None, None, mu_z, logvar_z

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu_f, logvar_f,mu_z, logvar_z, mu_prior_z=None, logvar_prior_z=None):
	#Loss = -ELBO = -E_q[log p(x | z,f)] + KL(q(f|x) | p(f)) + KL(q(z|x) | p(z))
	#embed()
	recon_x = torch.cat(recon_x, 0)
	MSE = 0.5*F.mse_loss(recon_x.squeeze(), x.squeeze(), reduction = 'sum')# MC estimate of -E_q[log p(x | z,f)]
	KL_f = -0.5 * torch.sum(1 + logvar_f - mu_f.pow(2) - logvar_f.exp()) # exact value of KL(q(f|x)|p(f))
	#print(KL_f.size())
	KL_z = 0 # MC estimate of KL(q(z|x) | p(z))
	for i in range(len(mu_z)):
		#mu_z[i] = mu_z[i].cuda()
		mu_z[i] = mu_z[i]
		#mu_prior_z[i] = mu_prior_z[i].cuda()
		
		#mu_diff = mu_z[i] - mu_prior_z[i]
		mu_diff = mu_z[i]
		#KL_z += -0.5 * torch.sum(1 + logvar_z[i].cuda() - logvar_prior_z[i].cuda() - logvar_z[i].cuda().exp()/logvar_prior_z[i].cuda().exp() - mu_diff.cuda().pow(2)/logvar_prior_z[i].cuda().exp())
		#KL_z += -0.5 * torch.sum(1 + logvar_z[i] - logvar_prior_z[i]
		#	- logvar_z[i].exp()/logvar_prior_z[i].exp() - mu_diff.pow(2)/logvar_prior_z[i].exp())
		KL_z += -0.5 * torch.sum(1 + logvar_z[i] - logvar_z[i].exp() - mu_diff.pow(2))

	return MSE + KL_f + KL_z

# train our model
def train(data, net, z_prior = False):
   
	start = time.time()

	optimizer = optim.Adam(net.parameters(), lr=1e-4)

	losses = []
	data = data.transpose(0,1)
	current = time.time()
	for epoch in range(200):
		total_loss = 0

		if epoch == 0:
			torch.save(net.state_dict(), 'SavedModels/train_epoch_%d.pt' % epoch)

		print(epoch)

		optimizer.zero_grad()
		
		recon_x, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z = net(data)
		# if not z_prior:
		# 	mu_prior_z = [torch.zeros(1, net.latent_dimz) for i in range(len(mu_z))]
		# 	logvar_prior_z = [torch.zeros(1, net.latent_dimz) for i in range(len(mu_z))]
		
		#loss = loss_function(recon_x, data, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z)
		loss = loss_function(recon_x, data, mu_f, logvar_f, mu_z, logvar_z)
		loss.backward()
		optimizer.step()

		total_loss += loss.data.item()
		print(total_loss/len(data))
		losses.append(total_loss/len(data))
		
		if (epoch + 1)%100 == 0:
			torch.save(net.state_dict(), 'SavedModels/train_epoch_%d.pt' % epoch)

	return losses


#data = [sample_data() for i in range(10)]
#net = SequentialVAE(2, 2, 2)
#losses = train(data, net)
#plt.plot(losses)
#plt.show()
#embed()

# load the image data 
# X_train/test stores the frames: (N_train, T, width, height, N_channel)
# A_train/test stores the labels of the attributes: (N_train, T, 4, 6)
# D_train/test stores the labels of the actions: (N_train, T, 9)
if __name__ == '__main__':
	X_train, X_test, A_train, A_test, D_train, D_test = sprites_act('', return_labels=True)

	X_train = torch.from_numpy(X_train)
	X_train = X_train.transpose(3,4).transpose(2,3) #hacky way to move the channel to the correct location


	X_train = X_train.to(device)

	#subsample to 100 points
	X_train = X_train.narrow(0, 0, 1)#.cuda() 

	latent_dimz = 32 # total of 9 different actions 
	latent_dimf = 256 # total number of types of characters
	net = CNNSequentialVAE(3, 64, 64, 8, latent_dimz, latent_dimf)
	net.to(device)
	torch.save(net.state_dict(), 'SavedModels/test_net.pt') #remeber to change this for different models
	losses = train(X_train, net)
	torch.save(net.state_dict(), 'SavedModels/post_train.pt') #remeber to change this for different models
	plt.plot(losses)
	plt.show()

