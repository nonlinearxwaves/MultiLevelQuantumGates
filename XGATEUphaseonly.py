# -*- coding: utf-8 -*-
"""

Solve by training for an X gate
case complex unitary matrix

now is not working, need to try with unitary matrix
recheck all and tests, to do...

22 september 2018

implement an X gate of size N
with a stopping criterion as the accuracy is below a threshold
allows to study the scaling with N

the training is done with a phase only diagonal matrix exp(i theta)

@author: nonli
"""

import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from stoppingRoutines import Stopper

#%%get hostname
import socket
print(socket.gethostname())

#%% to print matrix on screen with precision
import contextlib
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)

#%% learning rate
learning_rate=0.1
#%% threshold for stopping iterations in validation cost
threshold_valid=1e-3
#%% numer of dimensions
N=3
#%% number of ancillas
numberancilla=2
#%% number of training points
ntrain=100
nvalid=50
#%% epochs
epochs=1000
display_steps=2
strip_progress=display_steps
#%% file to save
now = datetime.now()
if socket.gethostname()=='hawaii':
    tensorboarddir = "/home/claudio/tensorflow_claudio_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
elif socket.gethostname()=='DESKTOP-667QSME':
    tensorboarddir = "c:/Users/claudio/tensorflow_claudio_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"    
else:
    tensorboarddir = "c:/Users/nonli/tensorflow_claudio_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
#%% random seed 
# to do, extract from now
timestamp = int(time.mktime(datetime.now().timetuple()))
RANDOM_SEED=timestamp
print('Random seed = ' + repr(timestamp))

#%% define graph
tf.reset_default_graph()

#%% saver
#global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
#saver=tf.train.Saver()


#%%
# seed random number generation
tf.set_random_seed(RANDOM_SEED)
np.random.seed(seed=RANDOM_SEED)


#%% arbitrary matrix with size X1
N1=N
dataXN=np.zeros((N1,N1),dtype=np.complex64)
for ix in range(N1):
    dataXN[ix,divmod(ix+1,N1)[1]]=1.0
dataXN_np=np.asarray(dataXN,dtype=np.complex64)
print('Target matrix')
print(dataXN_np)
XT=tf.constant(dataXN_np,dtype=tf.complex64)

#%% dimension del rigged space
N2=(numberancilla+1)*N

#%% random unitary matrix
#Ureal=tf.random_uniform([N, N])
#Uimag=tf.random_uniform([N, N])
#U=tf.complex(Ureal,Uimag)
#dataU=[[0.88901067, 0.8451152,  0.19780159,],
# [0.44519174, 0.36765718, 0.47530866],
# [0.7287617,  0.2972089,  0.24744892]]
dataUc=np.zeros([N2,N2],dtype=np.complex64)
for ix in range(N2):
    for iy in range(N2):
        dataUc[ix,iy]=np.random.random()+1j*np.random.random()
dataU_np=np.asarray(dataUc,dtype=np.complex64)
Q , _ = np.linalg.qr(dataU_np) #use LQ to generate a unitary matrix
#with printoptions(precision=2, suppress=True):
#    print("Structure Matrix")
#    print(Q)
#    print("Check unitarity of structure matrix Q^ Q")
#    print(np.inner(np.conj(Q),Q))
#U=tf.constant(dataU_np,dtype=tf.complex64)
U=tf.constant(Q,dtype=tf.complex64)
#TU=tf.complex(tf.transpose(Ureal),tf.transpose(Uimag))
#%% neutral matrxi
Neutral=tf.constant(np.zeros([N,N]),dtype=tf.float32)

#%% identity matrix
#IE=tf.eye(N,dtype=tf.complex64)

#%% tensore derivata di g rispetto a v (outerproduct)
#npI=np.eye(N)
#gklvij=np.multiply.outer(dataU_np,npI)
#dg=tf.constant(gklvij)
#dgtf=tf.einsum('ij,kl->ijkl',U,IE) #% questo non serve per equazioni lineari

#%% generate the input and the labels

weights0=tf.ones([N2,],dtype=tf.float32)
V0=tf.ones([N2,N2],dtype=tf.float32)
V0reduced=tf.ones([N,N],dtype=tf.float32)

VC=tf.ones([N2,N2],dtype=tf.complex64)


phase0=tf.get_variable("phases",initializer=weights0,dtype=tf.float32)

#Vreal=tf.get_variable("Vr",initializer=V0,dtype=tf.float32)
#Vimag=tf.get_variable("Vi",initializer=V0,dtype=tf.float32)
V=tf.get_variable("V",initializer=VC,dtype=tf.complex64,trainable=False)

#%% define ancilla
Nancilla=N*numberancilla

ar0=tf.ones([Nancilla,1],dtype=tf.float32)
ar=tf.get_variable("realancilla",initializer=ar0,dtype=tf.float32)
ai=tf.get_variable("imageancilla",initializer=ar0,dtype=tf.float32)


ancilla0=tf.ones([Nancilla,1],dtype=tf.complex64)
ancilla=tf.get_variable("cancilla",initializer=ancilla0,dtype=tf.complex64, trainable=False)



#%% transfer matrix
transfer_matrix=tf.get_variable("transfer_matrix",initializer=V0,trainable=False)
transfer_matrix_reduced=tf.get_variable("transfer_matrix_reduced",initializer=V0reduced,trainable=False)
#%% place holder
x=tf.placeholder(dtype=tf.complex64,shape=(N,1),name="x")
#y=tf.placeholder((N,1),dtype=tf.float32) #non serve qui y

#%% generate training set
xtrains=np.zeros((N,ntrain),dtype=np.complex64)
for j in range(ntrain):
   for i in range(N):
        xtrains[i,j]=np.random.random_sample()+1j*np.random.random_sample()
#%% generate validation set
xvalids=np.zeros((N,ntrain),dtype=np.complex64)
for j in range(nvalid):
   for i in range(N):
        xvalids[i,j]=np.random.random_sample()+1j*np.random.random_sample()
        
#%% equation
with tf.name_scope("equation") as scope:
    #target vector reduced
    yt=tf.matmul(XT,x)
    # costruisce il vettore nel rigged space    
    ancilla=tf.complex(ar,ai)
    xrigged=tf.concat([x, ancilla],0)
    #
    Vreal=tf.diag(tf.cos(phase0))
    Vimag=tf.diag(tf.sin(phase0))
    V=tf.complex(Vreal,Vimag)
    transfer_matrix=tf.matmul(U,V)
    # estrae qui solo le prime NxN
    yinternal=tf.matmul(transfer_matrix,xrigged)
    transfer_matrix_reduced=tf.slice(transfer_matrix,[0, 0],[N,N])
    yreduced=tf.slice(yinternal,[0 , 0],[N, 1])
    equation=yreduced-yt
    eqreal=tf.real(equation)
    eqimag=tf.imag(equation)
    cost_function=tf.reduce_mean(tf.square(eqreal)+
                                 tf.square(eqimag))
    tf.summary.scalar('cost_function',cost_function)
#%%TO DO : TRY OTHER MINIMIZER
with tf.name_scope("train") as scope:
#    global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
#           cost_function, global_step=global_step)    
 #   optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
#           cost_function, global_step=global_step)
   optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
           cost_function)

#%% writer
train_writer=tf.summary.FileWriter(tensorboarddir)
merged=tf.summary.merge_all()
#%% object that control the stop condition
stops=Stopper(strip_progress)
   
#%%
xtmp=np.zeros((N,1),dtype=np.float32)
with tf.Session()  as sess:
    sess.run(tf.global_variables_initializer())
    train_writer.add_graph(sess.graph)
    for epoch in range(epochs):
        avg_cost=0.
        for i in range(ntrain):
           xtmp=np.reshape(xtrains[0:N,i],(N,1))
    ##           xtmp=xtrains[0:N,i]
#           [xrig, x1, yrig, y1]=sess.run([xrigged, x, yinternal, yreduced],feed_dict={x: xtmp})
#           print(xrig)
#           print(x1)
#           print(yrig)
#           print(y1)
#           input("Press Any ...")
           sess.run(optimizer,feed_dict={x: xtmp})
           avg_cost+=sess.run(cost_function, feed_dict={x: xtmp})
           summary=sess.run(merged, feed_dict={x: xtmp}) 
           train_writer.add_summary(summary,i+epoch*epochs)
        avg_cost=avg_cost/N
        # store the training error in the shift register (for stopping condition, not used)
#        stops.insert(avg_cost)
        # messagers
        if epoch % display_steps == 0:
           # evaluate the validation error
           avg_cost_valid=0.
           for i in range(nvalid):
               xtmp_valid=np.reshape(xvalids[0:N,i],(N,1))
               avg_cost_valid+=sess.run(cost_function, feed_dict=
                                        {x: xtmp_valid})
           avg_cost_valid=avg_cost_valid/nvalid
           print('epoch '+repr(epoch))
           print('cost '+repr(avg_cost))
           print('valid cost '+repr(avg_cost_valid))
         # check the validation cost and if needed exit the iteration
        if avg_cost_valid < threshold_valid:
             print('Convergence in validation reacher at step '+ 
                   repr(epoch))
             break
    Tfinal=transfer_matrix_reduced.eval()
    phasefinal=phase0.eval()
    Vfinal=V.eval()
    ancillas=ancilla.eval()
#    print('Determinant Structure matrix ' + repr(np.linalg.det(dataU_np)))
        
#%%
sess.close()

eig1,_=np.linalg.eig(Vfinal)
        
with printoptions(precision=2, suppress=True):
#    print("Final V")
#    print(Vfinal)
#    print("Check unitarity of V ")
#    print(np.inner(np.conj(Vfinal),Vfinal))
    print("Final T")
    print(Tfinal)
    print("Check unitarity of T ")
    print(np.inner(np.conj(Tfinal),Tfinal))
    print("phase")
    print(phasefinal)
#    print("Eigenvalues of V")
#    print(eig1)
    print("Moduli of Eigenvalues of V")
    print(np.absolute(eig1))
    print("Phases of Eigenvalues of V (deg)")
    print(np.angle(eig1,deg=True))
    print("ancillas")
    print(ancillas)