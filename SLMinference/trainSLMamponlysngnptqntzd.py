#!/usr/bin/env python
# coding: utf-8

# Study the training with M of a 
# single qtrit gate by a SLM phase only with dimension N
#
# Version with a single input (plane wave)
# with size N, and padded with zero to size M
#
# The model of the SLM is phase only 
#
# By Claudio
# Initial version 31 Agoust 2019, single input training with clipping of real and imaginary part
# 



import sys
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt




# import utilities
#libPath='..\\..\\utilities\\'
libPath='../../utilities/'
if not libPath in sys.path: 
    sys.path.append(libPath)
  #   import syssys.path.append('../utilities')

from utilitiesquantumgates import SLM
from utilitiesquantumgates import quantumgates 
from utilitiesquantumgates import quantumgatesinference 
from utilitiesquantumgates import utilities
from tensorboardutilities import tensorboardutilities
from datetime import datetime
# datatypes
npdatatype=np.complex64
tfdatatype=tf.complex64
tfrealdatatype=tf.float32 # to use double switch aboe to complex128


#%% dimensions


N=3 # reduced dimensions
M=10 # embedding dimension


#%% target gate


X_np=quantumgates.Xgate(N,npdatatype)
utilities.printonscreennp(X_np)    
        




#%% Generate the random unitary matrix
U_np=quantumgates.randomU(M,npdatatype)  
#utilities.printonscreennp(U_np)    



#%% train amplitude only with -1.0 and 1.0 and zero image


out=\
    SLM.complexqtzd(
            X_np,U_np,
            verbose=2,
            epochs=1000,
            display_steps=50,
            realMAX=1.0,
            realMIN=-1.0,
            imagMAX=0.0,
            imagMIN=0.0,
            nbits=1
            )                                                                 
#%%
sys.exit()


#print(out)


# In[ ]:


# test the solution
#utilities.printonscreennp(Tinitial)
#utilities.printonscreennp(Sin)
#utilities.printonscreennp(np.matmul(U_np,Sin))
#utilities.printonscreennp(U_np)
#utilities.printonscreennp(Tfinal)


# In[ ]:

#get_ipython().run_cell_magic('javascript', '', 'require(\n    ["base/js/dialog"], \n    function(dialog) {\n        dialog.modal({\n\n                title: \'Notebook Halted\',\n                body: \'This notebook is no longer running; the kernel has been halted. Close the browser tab, or, to continue working, restart the kernel.\',\n                buttons: {\n                    \'Kernel restart\': { click: function(){ Jupyter.notebook.session.restart(); } }\n                }\n        });\n    }\n);\nJupyter.notebook.session.delete();')

#%%
# # Test training for N=3 and M=10 

# # Scaling of needed epochs with respect to M =redo with more points

#%%Scaling with M with 1 bit

Mmin=5
Mmax=100
step=5
Ms=list()
Nepoch=list()
Cost1=list()
count=0
for im in range(Mmin,Mmax+1,step):
    Ms.append(im)
    U_np=quantumgates.randomU(im,npdatatype)  
    out=\
    SLM.complexqtzd(
            X_np,U_np,
            verbose=1,
            epochs=1000,
            display_steps=50,
            realMAX=1.0,
            realMIN=-1.0,
            imagMAX=0.0,
            imagMIN=0.0,
            nbits=1
            )                                                                 
    Nepoch.append(out['epoch'])
    Cost1.append(out['cost'])
    count=count+1
    print('Running with M ' + 
          repr(im)+' cost  '+repr(out['cost']))
    
#Scaling with M with 8 bit

Ms=list()
Nepoch=list()
Cost8=list()
count=0
for im in range(Mmin,Mmax+1,step):
    Ms.append(im)
    U_np=quantumgates.randomU(im,npdatatype)  
    out=\
    SLM.complexqtzd(
            X_np,U_np,
            verbose=1,
            epochs=1000,
            display_steps=50,
            realMAX=1.0,
            realMIN=-1.0,
            imagMAX=0.0,
            imagMIN=0.0,
            nbits=8
            )                                                                 
    Nepoch.append(out['epoch'])
    Cost8.append(out['cost'])
    count=count+1
    print('Running with M ' + 
          repr(im)+' cost  '+repr(out['cost']))
    
#%%Scaling with M without quantization

Ms=list()
Nepoch=list()
Cost=list()
count=0
for im in range(Mmin,Mmax+1,step):
    Ms.append(im)
    U_np=quantumgates.randomU(im,npdatatype)  
    out=\
    SLM.complex(
            X_np,U_np,
            verbose=1,
            epochs=1000,
            display_steps=50,
            realMAX=1.0,
            realMIN=-1.0,
            imagMAX=0.0,
            imagMIN=0.0
            )                                                                 
    Nepoch.append(out['epoch'])
    Cost.append(out['cost'])
    count=count+1
    print('Running with M ' + 
          repr(im)+' cost  '+repr(out['cost']))



#%%
plt.figure(1)
#plt.subplot(311)
plt.plot(Ms,Cost1)
plt.plot(Ms,Cost8)
plt.plot(Ms,Cost)

plt.savefig('figurescaling2.eps', dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='eps',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)


# In[ ]:


print(Nepoch)


# In[ ]:


print(out)


# In[ ]:


print(Ms)


# In[ ]:


print(Nepoch)


# In[ ]:




