import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from IPython import embed
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
def train(data, net):
   
	start = time.time()

	optimizer = optim.Adam(net.parameters(), lr=1e-4)

	losses = []

	for epoch in range(10000):
		total_loss = 0
		for x in data:
			optimizer.zero_grad()
			recon_x, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z = net(x)
			loss = loss_function(recon_x, x, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z)
			loss.backward()
			optimizer.step()

			total_loss += loss.data.item()
		if (epoch + 1)%100 == 0:
			print(total_loss/len(data))
			losses.append(total_loss/len(data))

	return losses

#how our data is generated
def sample_data():
	sigma_z = 1
	sigma_x = 0.5
	lamb = 0.2
	w = torch.FloatTensor([[0.2,0.5],[-0.7,-0.3]])

	#latent variables
	z_1 = torch.randn(2,1)*sigma_z/math.sqrt(1 - lamb**2)
	z_2 = lamb*z_1 + torch.randn(2,1)*sigma_z

	x_1 = torch.mm(w,z_1) + torch.randn(2,1)*sigma_x
	x_2 = torch.mm(w,z_2) + torch.randn(2,1)*sigma_x

	return [x_1.view(-1,2), x_2.view(-1,2)]

data = [sample_data() for i in range(10)]
net = SequentialVAE(2, 2, 1)
losses = train(data, net)
plt.plot(losses)
plt.show()
embed()