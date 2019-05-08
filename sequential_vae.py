import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from IPython import embed
import time
import math
import matplotlib.pyplot as plt
import numpy as np

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

# our model
class SequentialVAE(nn.Module):
	def __init__(self, input_dim, latent_dimz, latent_dimf, sigma_f, sigma_z, sigma_x, lamb, w):
		super(SequentialVAE, self).__init__()

		#true parameter values
		self.sigma_f = torch.nn.Parameter(sigma_f, requires_grad=False)
		self.sigma_z = torch.nn.Parameter(sigma_z, requires_grad=False)
		self.sigma_x = torch.nn.Parameter(sigma_x, requires_grad=False)
		self.lamb = torch.nn.Parameter(lamb, requires_grad=False)
		self.w = torch.nn.Parameter(w, requires_grad=False)
		
		self.input_dim = input_dim # dimension of x
		self.latent_dimz = latent_dimz # dimension of z
		self.latent_dimf = latent_dimf # dimension of f

		# for the encoder for z, q(z_t|x_t)
		hidden_dim = 8
		self.fc1 = nn.Linear(self.input_dim, hidden_dim)
		self.fc21 = nn.Linear(hidden_dim, self.latent_dimz)
		self.fc22 = nn.Linear(hidden_dim, self.latent_dimz)

		# for the encoder for f, q(f|x_1:T)
		RNN_hidden_dim = 8
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
		hidden = self.RNN.initHidden(x[0].size()[0])
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

	def prior_fixed(self,z):
		mu_prior = [torch.zeros(1,2)]
		logvar_prior = [torch.log((self.sigma_z**2)*torch.ones(1,2)) for i in range(len(z))]#/(1 - self.lamb**2))]
		for i in range(0, len(z)-1):
		# 	#mu_prior.append(lamb*z[i])
		 	mu_prior.append(z[i])
		# 	logvar_prior.append(torch.log((self.sigma_z**2)*torch.ones(1,2)))
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

		#recon_x = [self.decode(z[i], f) for i in range(len(z))]
		recon_x = [torch.mm(z[i], self.w) + f for i in range(len(z))]
		mu_prior, logvar_prior = self.prior_fixed(z)
		return recon_x, mu_f, logvar_f, mu_z, logvar_z, mu_prior, logvar_prior


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

def vEM_loss(recon_x, x, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z, sigma_f, sigma_x):
	
	recon_x = torch.cat(recon_x, 0).view(3,-1, 2)
	MSE = 0.5*F.mse_loss(recon_x, x, reduction='sum')/sigma_x**2 + torch.log(2*math.pi*sigma_x**2)*recon_x.size()[0]*recon_x.size()[1]
	#embed()
	KL_f = -0.5 * torch.sum(1 + logvar_f - torch.log(sigma_f**2) - mu_f.pow(2)/sigma_f**2 - logvar_f.exp()/sigma_f**2)
	KL_z = 0
	for i in range(len(mu_z)):
		mu_diff = mu_z[i] - mu_prior_z[i]
		KL_z += -0.5 * torch.sum(1 + logvar_z[i] - logvar_prior_z[i] - logvar_z[i].exp()/logvar_prior_z[i].exp() - mu_diff.pow(2)/logvar_prior_z[i].exp())

	return MSE + KL_f + KL_z
# train our model
def train(data, net, z_prior = False):
   
	start = time.time()

	optimizer = optim.Adam(net.parameters(), lr=.01)

	losses = []

	for epoch in range(1000):
		total_loss = 0
		
		optimizer.zero_grad()
		recon_x, mu_f, logvar_f, mu_z, logvar_z, mu_prior_z, logvar_prior_z = net(data)
		loss = vEM_loss(recon_x, data, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z, net.sigma_f, net.sigma_x)
		loss.backward()
		optimizer.step()

		total_loss += loss.data.item()
		if (epoch + 1)%100 == 0:
			print(total_loss/len(data))
			losses.append(total_loss/len(data))

	true_loss = 0
	for i in range(1000):
		recon_x, mu_f, logvar_f, mu_z, logvar_z, mu_prior_z, logvar_prior_z = net(data)
		loss = vEM_loss(recon_x, data, mu_f, logvar_f, mu_prior_z, logvar_prior_z, mu_z, logvar_z, net.sigma_f, net.sigma_x)
		true_loss += loss/len(data)

	return true_loss/1000

#how our data is generated
def sample_data():

	#true parameter values
	sigma_f = 2.0
	sigma_z = 0.5
	sigma_x = 0.5
	#lamb = 0.2
	w = torch.FloatTensor([[1, 0],[0, 1]])

	#latent variables
	f = torch.randn(1,2)*sigma_f
	z_1 = torch.randn(1,2)*sigma_z#/math.sqrt(1 - lamb**2)
	#z_2 = lamb*z_1 + torch.randn(1,2)*sigma_z
	#z_3 = lamb*z_2 + torch.randn(1,2)*sigma_z
	z_2 = z_1 + torch.randn(1,2)*sigma_z
	z_3 = z_2 + torch.randn(1,2)*sigma_z

	x_1 = torch.mm(z_1,w) + f + torch.randn(1,2)*sigma_x
	x_2 = torch.mm(z_2,w) + f + torch.randn(1,2)*sigma_x
	x_3 = torch.mm(z_3,w) + f + torch.randn(1,2)*sigma_x

	return torch.cat([x_1, x_2, x_3], 0).view(3, 2)


sigma_f = torch.FloatTensor([2.0])
sigma_z = torch.FloatTensor([0.5])
#sigma_x = torch.FloatTensor([0.5])
lamb = torch.FloatTensor([0.2])
w = torch.FloatTensor([[1, 0],[0, 1]])

samples = 50
data = [sample_data() for i in range(samples)]
train_data = torch.cat(data,0).view(samples,3,2).transpose(0,1)
all_losses = []
all_ll = []

def log_likelihood(data, sigma_f, sigma_x, sigma_z):
	ll = 0
	S = np.array(
		[[sigma_f**2 + sigma_z**2 + sigma_x**2, 0, sigma_f**2 + sigma_z**2, 0, sigma_f**2 + sigma_z**2, 0],
		[0, sigma_f**2 + sigma_z**2 + sigma_x**2, 0, sigma_f**2 + sigma_z**2, 0, sigma_f**2 + sigma_z**2],
		[sigma_f**2 + sigma_z**2, 0, sigma_f**2 + 2*sigma_z**2 + sigma_x**2, 0, sigma_f**2 + 2*sigma_z**2, 0],
		[0, sigma_f**2 + sigma_z**2, 0, sigma_f**2 + 2*sigma_z**2 + sigma_x**2, 0, sigma_f**2 + 2*sigma_z**2],
		[sigma_f**2 + sigma_z**2, 0, sigma_f**2 + 2*sigma_z**2, 0, sigma_f**2 + 3*sigma_z**2 + sigma_x**2, 0],
		[0, sigma_f**2 + sigma_z**2, 0, sigma_f**2 + 2*sigma_z**2, 0, sigma_f**2 + 3*sigma_z**2 + sigma_x**2]])
	
	print(S)
	
	for x in data:
		x = x.view(6).numpy()
		ll += -0.5*( np.log(np.linalg.det(S)) + np.matmul(np.transpose(x),np.matmul(np.linalg.inv(S), x)) + 6*np.log(2*math.pi) )
		
	return ll/3

x_vals = [0.1*i for i in range(1,21)]
for x in x_vals:
	print("x = ", x)
	sigma_x = torch.FloatTensor([x])
	net = SequentialVAE(2, 2, 2, sigma_f, sigma_z, sigma_x, lamb, w)
	true_loss = train(train_data, net)
	
	all_losses.append(-true_loss.detach().numpy())
	all_ll.append(log_likelihood(data, sigma_f, sigma_x, sigma_z))
max_ll = [i for i in range(len(all_ll)) if all_ll[i] == max(all_ll)][0]
max_elbo = [i for i in range(len(all_losses)) if all_losses[i] == max(all_losses)][0]
plt.plot(x_vals, all_losses, color = 'b')
plt.plot(x_vals, all_ll, color = 'r')
print(plt.ylim())
plt.axvline(x_vals[max_ll], ymax = (all_ll[max_ll] - plt.ylim()[0])/(plt.ylim()[1] - plt.ylim()[0]), color = 'r')
plt.axvline(x_vals[max_elbo],  ymax = (all_losses[max_elbo]- plt.ylim()[0])/(plt.ylim()[1] - plt.ylim()[0]), color = 'b')
plt.show()
embed()