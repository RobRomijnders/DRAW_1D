# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 10:51:18 2016

@author: rob
"""
import numpy as np
import matplotlib.pyplot as plt

def clip(a,b=None):
  if b is None:
    return min(max(0,a),D-1)
  else:
    a = min(max(0,a),D-1)
    b = min(max(0,b),D-1)
    return range(a,b)

def sigmoid(value):
  return np.divide(1.0, 1+np.exp(-value))



def plot_DRAW_read(par_r,par_w,par_g, X ,direc = '/tmp/',Nplot = 5,cv=None):
  """Function to plot the read boxes for the DRAW model
  input:
  - par_r: the parameters of the read operation.
      expected list of length seq_len of np arrays in [batch_size,2]
      or expected np array in [seq_len, batch_size, 2]
  - par_w: the parameters of the write operation, like par_r
  - X: the input data in [batch_size, D]
  - direc: a directory where we can save consequetives canvases
  - Nplot: how many plots you want
  - cv: the canvases. If cv is None, we plot the digits as background"""
  global D
  batch_size,D = X.shape
  if not direc[-1] == '/':
    direc += '/'

  #Put all the parameters in a dictionary
  par = {}
  par['r'] = par_r
  par['w'] = par_w
  par['g'] = par_g
  for key in par:
    if isinstance(par[key],list):  #convert list to numpy array
      par[key] = np.stack(par[key])
    if not key == 'g':
      par[key] = par[key].astype(int)


  # Check if the input sizes match expected sizes
  _,_,write_size,Dg = par['g'].shape  #par['g'] now in [seq_len,batch_size,write_param,D]
  assert D == Dg, 'Your input data and write Gaussians have different signal length'
  seq_len = par['r'].shape[0]
  assert par['r'].shape[1] == batch_size, 'Your input data and parameters have different batch_size'
  if isinstance(cv,list):
    cv = np.stack(cv)  # in [seq_len, batch_size, D]
    assert seq_len == cv.shape[0], 'Your canvas size doesnt match your parameters'


  #Randomly pick some data to display
  im_ind = np.random.choice(batch_size,Nplot)

  for t in range(seq_len):
    f, axarr = plt.subplots(Nplot, len(par))
    imname = direc + 'canvas'+str(10+t)+'.png'  #Add 10, so ImageMagick does the sorting right. Otherwise 2 comes after 19
    for col,(key,value) in enumerate(par.iteritems()):
      for l in range(Nplot):
        if key == 'g': value = par['w']
        delta = value[t,im_ind[l],1]
        gx = value[t,im_ind[l],0]
        if key == 'r':
          axarr[l,col].plot(X[im_ind[l]])
        elif key == 'w':
          axarr[l,col].plot(cv[t,im_ind[l],:])
        elif key == 'g':
          for wr in xrange(write_size):
            axarr[l,col].plot(par[key][t,im_ind[l],wr,:],'b')
          axarr[l,col].plot(np.sum(par[key][t,im_ind[l],:,:],0),'r')   #Plot the total sum of the Gaussians
        axarr[l,col].set_ylim([-3,3])
        if key == 'g': axarr[l,col].set_ylim([-1,1])
        axarr[l,col].set_xlim([0,D])
        x_bar = np.array([clip(gx-delta),clip(gx+delta)])
        y_bar = np.array([1.0, 1.0])
        axarr[l,col].plot(x_bar,y_bar)
        if not l == Nplot-1:
          plt.setp([axarr[l,col].get_xticklabels()], visible=False)
        if not col == 0:
          plt.setp([axarr[l,col].get_yticklabels()], visible=False)
    f.subplots_adjust(hspace=0)  #No horizontal space between subplots
    f.subplots_adjust(wspace=0)  #No vertical space between subplots
    plt.savefig(imname)

  #Save another five times for the .gif
  for tt in xrange(5):
    imname = direc + 'canvas'+str(10+t+tt)+'.png'
    plt.savefig(imname)
  plt.close('all')
  return




