# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders
"""
import sys
import socket

if 'rob-laptop' in socket.gethostname():
  sys.path.append('/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015/')
  sys.path.append('/home/rob/Dropbox/ml_projects/DRAW_1D/')
  direc = '/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015'  #Location of the UCR database
elif 'rob-com' in socket.gethostname():
  sys.path.append('/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015/')
  sys.path.append('/home/rob/Documents/DRAW_1D/')
  direc = '/home/rob/Documents/LSTM/UCR_TS_Archive_2015'  #Location of the UCR database

import numpy as np
import tensorflow as tf
from plot_DRAW_1D import *
import matplotlib.pyplot as plt
from tensorflow.models.rnn.rnn_cell import LSTMCell



"""Hyperparameters"""
hidden_size_enc = 256       # hidden size of the encoder
hidden_size_dec = 256       # hidden size of the decoder
patch_read = 5              # Size of patch to read
patch_write = 5             # Size of patch to write
read_size = 2*patch_read
write_size = patch_write
num_l=10                    # Dimensionality of the latent space
sl=10                       # Sequence length
batch_size=100
max_iterations=1000
learning_rate=1e-3
eps=1e-8                    # Small number to prevent numerical instability


REUSE_T=None               # indicator to reuse variables. See comments below
read_params = []           # A list where we save al read paramaters trhoughout code. Allows for nice visualizations at the end
write_params = []          # A list to save write parameters
write_gaussian = []           # A list to save Gaussian parameters for writing
Nplot = 5
ratio_train = 0.8

"""Load the data"""
# NonInvasiveFatalECG_Thorax2
# StarLightCurves
dataset = "ECG5000"
datadir = direc + '/' + dataset + '/' + dataset
data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
data_test_val = np.loadtxt(datadir+'_TEST',delimiter=',')[:-1]
data = np.concatenate((data_train,data_test_val),axis=0)

N,D = data.shape

ind_cut = int(ratio_train*N)

# Usually, the first column contains the target labels
X_train = data[:ind_cut,1:]
X_val = data[ind_cut:,1:]
N = X_train.shape[0]
Nval = X_val.shape[0]
D = X_train.shape[1]
y_train = data[:ind_cut,0]
y_val = data[ind_cut:,0]
print('We have %s observations with %s dimensions'%(N,D))

# Organize the classes
num_classes = len(np.unique(y_train))
base = np.min(y_train)  #Check if data is 0-based
if base != 0:
  y_train -=base
  y_val -= base
y_train = y_train.astype(int)
y_val = y_val.astype(int)


if True:  #Set true if you want to visualize the actual time-series
  f, axarr = plt.subplots(Nplot, num_classes)
  for c in np.unique(y_train):    #Loops over classes, plot as columns
    ind = np.where(y_train == c)
    ind_plot = np.random.choice(ind[0],size=Nplot)
    for n in xrange(Nplot):  #Loops over rows
      axarr[n,c].plot(X_train[ind_plot[n],:])
      # Only shops axes for bottom row and left column
      if not n == Nplot-1:
        plt.setp([axarr[n,c].get_xticklabels()], visible=False)
      if not c == 0:
        plt.setp([axarr[n,c].get_yticklabels()], visible=False)
  f.subplots_adjust(hspace=0)  #No horizontal space between subplots
  f.subplots_adjust(wspace=0)  #No vertical space between subplots
  plt.show()

cv_size = D                 # canvas size

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))


def filterbank(gx, sigma2,delta, N):
  # gx in [batch_size,1]
  # delta in [batch_size,1]
  # grid in [1,patch_read]
  # x_mu in [batch_size,patch_read]
  # x_mu in [batch_size,patch_read,1]
  # Fx in [batch_size, patch_read, D]
  # a in [1,1,D]
  a = tf.reshape(tf.cast(tf.range(D),tf.float32),[1,1,-1])
  grid = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
  x_mu = gx+(grid-N/2-0.5)*delta # eq 19
  x_mu = tf.reshape(x_mu,[-1,N,1])
  sigma2 = tf.reshape(sigma2,[-1,1,1])
  Fx = tf.exp(-tf.square((x_mu-a)/(-2*sigma2)))
  Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
  return Fx

def linear(x,output_dim):
  """ Function to compute linear transforms
  when x is in [batch_size, some_dim] then output will be in [batch_size, output_dim]
  Be sure to use right variable scope n calling this function
  """
  w=tf.get_variable("w", [x.get_shape()[1], output_dim])
  b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
  return tf.nn.xw_plus_b(x,w,b)


def attn_window(scope,h_dec,N):
  with tf.variable_scope(scope,reuse=REUSE_T):
    params=linear(h_dec,4)
  gx_,log_sigma2,log_delta,log_gamma=tf.split(1,4,params)  #eq (21)
  gx=(D+1)/2*(gx_+1)  #eq (22)
  sigma2=tf.exp(log_sigma2)
  delta=(D-1)/(N-1)*tf.exp(log_delta) # batch x N    #eq (24)
  Fx = filterbank(gx,sigma2,delta,N)    #eq (25,26)
  gamma = tf.exp(log_gamma)
  if scope == 'read': read_params.append(tf.concat(1,[gx,delta]))
  if scope == 'write': write_params.append(tf.concat(1,[gx,delta]))
  return (Fx,gamma)


def read(x,x_hat,h_dec_prev):
  """Function to implement eq 27"""
  Fx,gamma=attn_window("read",h_dec_prev,patch_read)
  # gamma in [batch_size,1]
  # Fx in [batch_size, patch_read, D]
  def filter_img(ecg,Fx,gamma,N):
    # ecg in [batch_size, D]
    ecg = tf.expand_dims(ecg,1)  # in [batch_size, 1, D]
    Fxt=tf.transpose(Fx,perm=[0,2,1])  # Fxt in [batch_size,D,patch_read]
    glimpse=tf.batch_matmul(ecg,Fxt) #in [batch_size, patch_read]
    glimpse = tf.squeeze(glimpse, [1])
    return glimpse*tf.reshape(gamma,[-1,1])
  x=filter_img(x,Fx,gamma,patch_read) # batch x (patch_read*patch_read)
  x_hat=filter_img(x_hat,Fx,gamma,patch_read)
  # x in [batch_size, patch_read]
  # x_hat in [batch_size, patch_read]
  return tf.concat(1,[x,x_hat]) # concat along feature axis in [batch_size, 2*patch_read]


def encode(state,input):
  #Run one step of the encoder
  with tf.variable_scope("encoder",reuse=REUSE_T):
    return lstm_enc(input,state)

def decode(state,input):
  #Run one step of the decoder
  with tf.variable_scope("decoder",reuse=REUSE_T):
      return lstm_dec(input, state)

def sample_lat(h_enc):
  """Sample in the latent space, using the reparametrization trick
  """
  #Mind the variable scopes, so that linear() uses the right Tensors
  with tf.variable_scope("mu",reuse=REUSE_T):
    mu=linear(h_enc,num_l)
  with tf.variable_scope("sigma",reuse=REUSE_T):
    logsigma=linear(h_enc,num_l)
    sigma=tf.exp(logsigma)
  return (mu + sigma*e, mu, logsigma, sigma)


def write(h_dec):
  """Function to implement 29"""
  with tf.variable_scope("writeW",reuse=REUSE_T):
      w=linear(h_dec,write_size) # [batch, patch_write]
  Fx,gamma=attn_window("write",h_dec,patch_write)
  wg = tf.mul(1.0/gamma,w)  #w times gamma
  wg = tf.expand_dims(wg,2)  # in [batch_size,patch_write,1]
  wr=tf.mul(wg,Fx)  # in [batch_size, D]
  write_gaussian.append(wr)
  return tf.reduce_sum(wr,1)


with tf.variable_scope("placeholders") as scope:
  x = tf.placeholder(tf.float32,shape=(batch_size,D)) # input [batch_size, D]
  e = tf.random_normal((batch_size,num_l), mean=0, stddev=1) # Qsampler noise
  lstm_enc = LSTMCell(hidden_size_enc, read_size+hidden_size_dec) # encoder Op
  lstm_dec = LSTMCell(hidden_size_dec, num_l) # decoder Op

with tf.variable_scope("States") as scope:
  canvas=[0]*sl # The canves gets sequentiall painted on
  mus,logsigmas,sigmas=[0]*sl,[0]*sl,[0]*sl # parameters for the Gaussian
  # Initialize the states
  h_dec_prev=tf.zeros((batch_size,hidden_size_dec))
  enc_state=lstm_enc.zero_state(batch_size, tf.float32)
  dec_state=lstm_dec.zero_state(batch_size, tf.float32)

with tf.variable_scope("DRAW") as scope:
  #Unroll computation graph
  for t in range(sl):
    c_prev = tf.zeros((batch_size,D)) if t==0 else canvas[t-1]
    x_hat=x-tf.sigmoid(c_prev) # error image
    r=read(x,x_hat,h_dec_prev)
    h_enc,enc_state=encode(enc_state,tf.concat(1,[r,h_dec_prev]))
    z,mus[t],logsigmas[t],sigmas[t]=sample_lat(h_enc)
    h_dec,dec_state=decode(dec_state,z)
    canvas[t]=c_prev+write(h_dec) # store results
    h_dec_prev=h_dec
    REUSE_T=True # Comment on REUSE_T below
#REUSE_T is initialized as None/False. That way, all the get_variable() functions initialize
# a new parameter. After we run the above for-loop once, we'd like the Computation graph to
# to share the matrices in the LSTM's. Therefore we set REUSE_T=True after the first loop


with tf.variable_scope("Loss_comp") as scope:
  #For now define a uncorrelated Gaussian as posteriour over the data.
  # for now, take every sample to be unit variance. Lateer we might have the network predict the variance too
  def gaussian_cost(t,o):
    s = 1.0  #For now take unit variance
    norm = tf.sub(o,t)
    z = tf.square(tf.div(norm, s))
    result = tf.exp(tf.div(-z,2.0))
    denom = 2.0*np.pi*s
    p = tf.div(result, denom)
    return -tf.log(p)
  x_recons=canvas[-1]  #canvas[-1] is the final canvas

  cost_recon=tf.reduce_sum(gaussian_cost(x,x_recons),1) # Bernoulli cost for reconstructed imae with true image as mean
  cost_recon=tf.reduce_mean(cost_recon)

  kls=[0]*sl  #list with saving the KL divergences
  for t in range(sl):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kls[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma-1,1)  #-sl*.5
  kl=tf.add_n(kls) # adds tensors over the list
  cost_lat=tf.reduce_mean(kl) # average over minibatches
  cost=cost_recon+cost_lat

with tf.variable_scope("Optimization") as scope:
  optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
  grads=optimizer.compute_gradients(cost)
  for i,(g,v) in enumerate(grads):  #g = gradient, v = variable
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # Clip the gradients of LSTM
  train_op=optimizer.apply_gradients(grads)

print('Finished comp graph')


fetches=[cost_recon,cost_lat,train_op]
costs_recon=[0]*max_iterations
costs_lat=[0]*max_iterations

sess=tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(max_iterations):
  batch_ind = np.random.choice(N,batch_size,replace=False)
  xtrain = X_train[batch_ind]
  feed_dict={x:xtrain}
  results=sess.run(fetches,feed_dict)
  costs_recon[i],costs_lat[i],_=results
  if i%100==0:
    print("iter=%d : cost_recon: %f cost_lat: %f" % (i,costs_recon[i],costs_lat[i]))

read_params_fetch = sess.run(read_params,feed_dict) #List of length (sl) with np arrays in [batch_size,2]
write_params_fetch = sess.run(write_params,feed_dict)
write_gaussian_fetch = sess.run(write_gaussian,feed_dict)
canvases = sess.run(canvas,feed_dict)
direc_plot = '/home/rob/Dropbox/ml_projects/DRAW_1D/canvas/'
plot_DRAW_read(read_params_fetch,write_params_fetch,write_gaussian_fetch, feed_dict[x],direc_plot,5,canvases)
#Now go to the directory and run  (after install ImageMagick)
#  convert -delay 20 -loop 0 *.png mnist.gif



