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

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)

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
		self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=(4, 4), stride=2)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2)	
		
		#for the encoder for z, q(z_t|x_t)
		self.fc11 = nn.Linear(in_features=256*2*2, out_features=self.latent_dimz)
		self.fc12 = nn.Linear(in_features=256*2*2, out_features=self.latent_dimz)

		#for the encoder for f, q(f|x_1:T)
		self.fc21 = nn.Linear(in_features=256, out_features=self.latent_dimf)
		self.fc22 = nn.Linear(in_features=256, out_features=self.latent_dimf)
		RNN_hidden_dim = 3
		# self.RNN = RNN(3* self.x_height * self.x_width, RNN_hidden_dim, 2*self.latent_dimf)
		self.RNN = RNN(256*2*2, RNN_hidden_dim, 2*self.latent_dimf) # for debugging 
		
		# for the decoder
		# self.fc0 = nn.Linear(in_features=self.latent_dimz, out_features = 256*30*30)
		self.fc1 = nn.Linear(in_features=self.latent_dimz+self.latent_dimf, out_features=256*2*2)
		self.conv_t1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2)
		self.conv_t2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2)
		self.conv_t3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=2)
		self.conv_t4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=2)

		self.sigmoid = nn.Sigmoid()

		# for the prior p(z_t|z_<t)
		RNN_hidden_dim2 = 3
		self.RNN_prior = RNN(self.latent_dimz, RNN_hidden_dim2, 2*self.latent_dimz)

	# compute parameters for q(z_t | x_t)
	def encode_z(self, x):
		h = x
		h = h.view(-1, self.image_channels, self.x_height, self.x_width)
		h = F.elu(self.conv1(h))
		h = F.elu(self.conv2(h))
		h = F.elu(self.conv3(h))
		h = F.elu(self.conv4(h))
		h = h.view(1, -1)
		mu = self.fc11(h)
		logvar = self.fc12(h)
		return mu, logvar
				
	
	# compute parameters for q(f | x[1:T])
	def encode_f(self, x):
		hidden = self.RNN.initHidden()
		for i in range(self.frames):
			h = x[i]
			h = h.view(-1, self.image_channels, self.x_height, self.x_width)
			h = F.elu(self.conv1(h))
			h = F.elu(self.conv2(h))
			h = F.elu(self.conv3(h))
			h = F.elu(self.conv4(h))
			#h = h.view(-1, 3* self.x_height * self.x_width) #this might be wrong
			h = h.view(1, -1) # this might be wrong
			output, hidden = self.RNN(h, hidden)
		mu, logvar = torch.split(output, self.latent_dimf, dim=1)
		return mu, logvar

	def prior(self, z):
		hidden = self.RNN_prior.initHidden()
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
		x = F.elu(self.fc1(combined))
		x = x.view(-1, 256, 2, 2)
		x = F.elu(self.conv_t1(x))
		x = F.elu(self.conv_t2(x))
		x = F.elu(self.conv_t3(x))
		x = F.elu(self.conv_t4(x))
		x = F.elu(self.sigmoid(x))
		return x

	# forward pass of the algorithm 
    # returns:
    #       - recon_x: the outputted value of the VAE
    #       - mu_f, logvar_f: parameters for q(f | x_1:T), a Gaussian
    #       - mu_prior, logvar_prior: parameters for p(z_t|z_<t), a Gaussian
    #       - mu_z, logvar_z: parameters for q(z_t | x_t), a Gaussian
	def forward(self, x):
		mu_f, logvar_f = self.encode_f(x)
		f = self.reparameterize(mu_f, logvar_f)
		z = []
		mu_z = []
		logvar_z = []
		for i in range(self.frames):
			mu_z_i, logvar_z_i = self.encode_z(x[i])
			z.append(self.reparameterize(mu_z_i,logvar_z_i))
			mu_z.append(mu_z_i)
			logvar_z.append(logvar_z_i)

		recon_x = [self.decode(z[i], f) for i in range(len(z))]
		mu_prior, logvar_prior = self.prior(z)
		return recon_x, mu_f, logvar_f, mu_prior, logvar_prior, mu_z, logvar_z

# our model
class SequentialVAE(nn.Module):
	def __init__(self, input_dim, latent_dimz, latent_dimf):
		super(SequentialVAE, self).__init__()
		
		self.input_dim = input_dim # dimension of x
		self.latent_dimz = latent_dimz # dimension of z
		self.latent_dimf = latent_dimf # dimension of f

		# for the encoder for z, q(z_t|x_t)
		hidden_dim = 3
		self.fc1 = nn.Linear(self.input_dim, hidden_dim)
		self.fc21 = nn.Linear(hidden_dim, self.latent_dimz)
		self.fc22 = nn.Linear(hidden_dim, self.latent_dimz)

		# for the encoder for f, q(f|x_1:T)
		RNN_hidden_dim = 3
		self.RNN = RNN(self.input_dim, RNN_hidden_dim, 2*self.latent_dimf)

		# for the decoder p(x_t|z_t,f)
		hidden_dim2 = 3
		self.fc3 = nn.Linear(self.latent_dimz + self.latent_dimf, hidden_dim2)
		self.fc4 = nn.Linear(hidden_dim2, self.input_dim)

		# for the prior p(z_t|z_<t)
		RNN_hidden_dim2 = 3
		self.RNN_prior = RNN(self.latent_dimz, RNN_hidden_dim2, 2*self.latent_dimz)

	# compute parameters for q(z_t | x_t)
	def encode_z(self, x):
		h1 = F.relu(self.fc1(x))
		mu = self.fc21(h1)
		logvar = self.fc22(h1)
		return mu, logvar

	# compute parameters for q(f|x_1:T)
	def encode_f(self, x):
		hidden = self.RNN.initHidden()
		for i in range(len(x)):
			output, hidden = self.RNN(x[i], hidden)
		mu, logvar = torch.split(output, self.latent_dimf, dim=1)
		return mu, logvar

	# compute parameters for p(z_t|z_<t)
	def prior(self, z):
		hidden = self.RNN_prior.initHidden()
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
		h1 = F.relu(self.fc3(combined))
		x = self.fc4(h1)
		return x

	# forward pass of the algorithm 
	# returns:
	# 	- recon_x: the outputted value of the VAE
	#	- mu_f, logvar_f: parameters for q(f | x_1:T), a Gaussian
	#	- mu_prior, logvar_prior: parameters for p(z_t|z_<t), a Gaussian
	#	- mu_z, logvar_z: parameters for q(z_t | x_t), a Gaussian
	def forward(self, x):
		mu_f, logvar_f = self.encode_f(x)
		f = self.reparameterize(mu_f, logvar_f)
		z = []
		mu_z = []
		logvar_z = []
		for i in range(len(x)):
			mu_z_i, logvar_z_i = self.encode_z(x[i])
			z.append(self.reparameterize(mu_z_i,logvar_z_i))
			mu_z.append(mu_z_i)
			logvar_z.append(logvar_z_i)

		recon_x = [self.decode(z[i], f) for i in range(len(z))]
		mu_prior, logvar_prior = self.prior(z)
		return recon_x, mu_f, logvar_f, mu_prior, logvar_prior, mu_z, logvar_z


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z):
	#Loss = -ELBO = -E_q[log p(x | z,f)] + KL(q(f|x) | p(f)) + KL(q(z|x) | p(z))
	MSE = sum([0.5*F.mse_loss(recon_x[i], x[i]) for i in range(len(x))]) # MC estimate of -E_q[log p(x | z,f)]
	KL_f = -0.5 * torch.sum(1 + logvar_f - mu_f.pow(2) - logvar_f.exp()) # exact value of KL(q(f|x)|p(f))
	KL_z = 0 # MC estimate of KL(q(z|x) | p(z))
	for i in range(len(mu_z)):
		mu_diff = mu_z[i] - mu_prior_z[i]
		KL_z += -0.5 * torch.sum(1 + logvar_z[i] - logvar_prior_z[i] - logvar_z[i].exp()/logvar_prior_z[i].exp() - mu_diff.pow(2)/logvar_prior_z[i].exp())

	return MSE + KL_f + KL_z

# train our model
def train(data, net, z_prior = False):
   
	start = time.time()

	optimizer = optim.Adam(net.parameters(), lr=1e-4)

	losses = []

	for epoch in range(10000):
		total_loss = 0
		for x in data:
			optimizer.zero_grad()
			recon_x, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z = net(x)
			if not z_prior:
				mu_prior_z = [torch.zeros(1, net.latent_dimz) for i in range(len(mu_z))]
				logvar_prior_z = [torch.zeros(1, net.latent_dimz) for i in range(len(mu_z))]
			loss = loss_function(recon_x, x, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z)
			loss.backward()
			optimizer.step()

			total_loss += loss.data.item()
		if (epoch + 1)%100 == 0:
			print(total_loss/len(data))
			losses.append(total_loss/len(data))

	return losses

#how our data is generated
# def sample_data():
# 	sigma_z = 1
# 	sigma_x = 0.5
# 	lamb = 0.2
# 	w = torch.FloatTensor([[0.2,0.5],[-0.7,-0.3]])

# 	#latent variables
# 	z_1 = torch.randn(2,1)*sigma_z/math.sqrt(1 - lamb**2)
# 	z_2 = lamb*z_1 + torch.randn(2,1)*sigma_z

# 	x_1 = torch.mm(w,z_1) + torch.randn(2,1)*sigma_x
# 	x_2 = torch.mm(w,z_2) + torch.randn(2,1)*sigma_x

# 	return [x_1.view(-1,2), x_2.view(-1,2)]

def sample_data():
	sigma_f = 2.0
	sigma_z = 1.0
	sigma_x = 0.5

	w = torch.FloatTensor([[0.2,0.5],[-0.7,-0.3]])

	f = torch.randn(1,2)*sigma_f

	z = [torch.randn(1,2)*sigma_z for i in range(3)]

	x = [f + torch.mm(i,w) + torch.randn(1,2)*sigma_x for i in z]

	return x


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
X_train, X_test, A_train, A_test, D_train, D_test = sprites_act('', return_labels=True)

X_train = torch.from_numpy(X_train)
X_train = X_train.transpose(3,4).transpose(2,3) #hacky way to move the channel to the correct location
latent_dimz = 9 # total of 9 different actions 
latent_dimf = 1296 # total number of types of characters
net = CNNSequentialVAE(3, 64, 64, 8, latent_dimz, latent_dimf)
losses = train(X_train, net)
plt.plot(losses)
plt.show()
embed()
