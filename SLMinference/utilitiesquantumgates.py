# -*- coding: utf-8 -*-
# Library of classes for varius quantum gates
#
# By Claudio
# Initial version september 2018
# Current version 3 february 2019 


import numpy as np
import tensorflow as tf

#class for defining various operators
class quantumgates:
    
    
    def Rz(a,N=2,datatype=np.complex64):
        ''' Return the 2x2 rotator matrix Rz
        
        Rz(alpha)=[exp(i a/2) 0; 0 exp(-i a /2)]
        
        see PRA 52, 3457 (1995)
        
        '''
        if N!=2:
            print("error: Rx only returned for N=2")
            return
        data=np.zeros((N,N),dtype=datatype)
        data[0,0]=np.exp(1j*0.5*a)
        data[0,1]=0.0
        data[1,0]=0.0
        data[1,1]=np.exp(-1j*0.5*a)
        return data

    def Ry(t,N=2,datatype=np.complex64):
        ''' Return the 2x2 rotator matrix Ry
        
        Ry(t)=[cos(t/2)  sin(t/2); -sin(t/2) cos(t/2)]
        
        see PRA 52, 3457 (1995)
        
        '''
        if N!=2:
            print("error: Ry only returned for N=2")
            return
        data=np.zeros((N,N),dtype=datatype)
        data[0,0]=np.cos(t*0.5)
        data[0,1]=np.sin(t*0.5)
        data[1,0]=-np.sin(t*0.5)
        data[1,1]=np.cos(t*0.5)
        return data

    def paulix(N=2,datatype=np.complex64):
        ''' Return the 2x2 Pauli matrix \sigma_x
        
        sigmax=[0 1; 1 0]
        
        see PRA 52, 3457 (1995)
        
        '''
        if N!=2:
            print("error: sigmax only returned for N=2")
            return
        data=np.zeros((N,N),dtype=datatype)
        data[0,0]=0.0
        data[0,1]=1.0
        data[1,0]=1.0
        data[1,1]=0.0
        return data

    
    def Xgate(N,datatype=np.complex64):
        #Return a matrix NxN that represent a X gate for 1 qdit with size N
        # in the input one has the size N and the optional datatype argument
        # default for datatype is np.complex64        
        dataXN=np.zeros((N,N),dtype=datatype)
        for ix in range(N):
            dataXN[ix,divmod(ix+1,N)[1]]=1.0
        return np.asarray(dataXN,dtype=datatype)
    
    def Zgate(N,datatype=np.complex64):
        #Return a matrix NxN that represent a Z gate for 1 qdit with size N
        # in the input one has the size N and the optional datatype argument
        # default for datatype is np.complex64        
        dataXN=np.zeros((N,N),dtype=np.complex64)
        for ix in range(N):
            dataXN[ix,ix]=np.complex(np.exp(1j*np.pi*ix/N))
        return np.asarray(dataXN,dtype=np.complex64)
            
    def randomU(N,datatype=np.complex64):
        #Return a random unitary matrix
        # in the input one has the size N and the optional datatype argument
        # default for datatype is np.complex64
        # Important need to set the randomseed
        dataUc=np.zeros([N,N],dtype=datatype)
        for ix in range(N):
            for iy in range(N):
                dataUc[ix,iy]=np.random.random()+1j*np.random.random()
        dataU_np=np.asarray(dataUc,dtype=datatype)
        Q , _ = np.linalg.qr(dataU_np) #use LQ to generate a unitary matrix  
        return Q

    def randomvector(N,datatype=np.complex64):
        #Return a random vector with complex values
        data=np.random.random([N,1])+1j*np.random.random([N,1])
        return data
    
    def unitarize(M_np):
        # Use LQ decomposition and return Q to convert a matrix into a unitary
        Q , _ = np.linalg.qr(np.asarray(M_np)) #use LQ to generate a unitary matrix  
        return Q        
      
    def randomuniform(N,datatype=np.complex64):
        #Return a random square matrix with uniform numbers
        # in the input one has the size N and the optional datatype argument
        # default for datatype is np.complex64
        # Important need to set the randomseed
        dataUc=np.zeros([N,N],dtype=datatype)
        for ix in range(N):
            for iy in range(N):
                dataUc[ix,iy]=np.random.random()+1j*np.random.random()
        return np.asarray(dataUc,dtype=datatype) 
    
    def randomdiagonal(N,datatype=np.complex64):
        #Return a random square matrix with uniform numbers
        # in the input one has the size N and the optional datatype argument
        # default for datatype is np.complex64
        # Important need to set the randomseed
        dataUc=np.zeros([N,N],dtype=datatype)
        for ix in range(N):
                dataUc[ix,ix]=np.random.random()+1j*np.random.random()
        return np.asarray(dataUc,dtype=datatype) 
                                
    def projector(N,M,datatype=np.complex64):
      #Return a projector, i.e., a matrix with size NxM formed by [IN, 0] 
      # with IN the identity matrix and all other elements zero
      data=np.zeros([N,M],dtype=datatype)
      if M<N:
        print("Error ! M cannot be smaller than N in quantumgates.projector")
        return 0.0
      for ix in range(N):
        data[ix,ix]=1.0
      return data
    
    def riggoperator(X,M,datatype=np.complex64):
    # given an operator X with size NxN return a rigged operator with size NxM
    # with shape [X | 0 ] with 0 a zeros matrix with size NxM-N
      N=X.shape[0]
      N1=X.shape[1]
      if N!=N1:
        print("Error ! X must be a square matrix in quantumgates.riggoperator")
        return 0.0        
      data=np.zeros([N,M],dtype=datatype)
      if M<N:
        print("Error ! M cannot be smaller than N in quantumgates.riggoperator")
        return data
      for ix in range(N):
        for iy in range(N):
          data[ix,iy]=X[ix,iy]
      return data
    
        
    def riggunitary(X,M,datatype=np.complex64):
    # given an operator X with size NxN return a rigged operator with size MxM
    # with shape [X | 0 ; 0^T | U(M-N)] with 0 a zeros matrix with size NxM-N
    # and U a random unitary matrix
      N=X.shape[0]
      N1=X.shape[1]
      if N!=N1:
        print("Error ! X must be a square matrix in quantumgates.riggunitary")
        return 0.0        
      data=np.zeros([M,M],dtype=datatype)
      if M<N:
        print("Error ! M cannot be smaller than N in quantumgates.riggunitary")
        return data
      for ix in range(N):
        for iy in range(N):
          data[ix,iy]=X[ix,iy]
      if M>N:    
          Ureduced=quantumgates.randomU(M-N,datatype)
          for ix in range(M-N):
              for iy in range(M-N):
                  data[ix+N,iy+N]=Ureduced[ix,iy]
      return data

    def riggzero(X,M,datatype=np.complex64):
    # given an operator X with size NxN return a rigged operator with size MxM
    # with shape [X | 0 ; 0^T | 0C)] with 0 a zeros matrix with size NxM-N
    # and 0C a zero matrix of size (M-N x M-N)
      N=X.shape[0]
      N1=X.shape[1]
      if N!=N1:
        print("Error ! X must be a square matrix in quantumgates.riggunitary")
        return 0.0        
      data=np.zeros([M,M],dtype=datatype)
      if M<N:
        print("Error ! M cannot be smaller than N in quantumgates.riggunitary")
        return data
      for ix in range(N):
        for iy in range(N):
          data[ix,iy]=X[ix,iy]
      if M>N:    
          for ix in range(M-N):
              for iy in range(M-N):
                  data[ix+N,iy+N]=0.0
      return data


  
    def riggidentity(X,datatype=np.complex64):
    # given an operator X with size NxN return a N+1xN+1 operator
    # with structure [1 | 0; 0 | X] i.e. embed in a identiy matrix
        X=np.matrix(X)
        (N,M)=X.shape
        if M!=N:
            print("Error: quantumgates.riggidentity only works with square matrix")
            return 0.0
        N1=N+1
        I1=np.eye(N1,dtype=datatype)
        for ix in range(N):
            for iy in range(N):
                I1[ix+1,iy+1]=X[ix,iy]
        return I1

    def multiriggidentity(X,N,datatype=np.complex64):
    # given an operator rigg many times the X as in riggunitary as far as 
    # the dimension is N
        X=np.matrix(X)
        (NX,MX)=X.shape
        if MX!=NX:
            print("Error: quantumgates.multiriggidentity only works with square matrix")
            return 0.0
        if N<MX:
            print("Warning: quantumgates.multiriggidentity, operator has size greater than N")
            return X
        tildeX=X
        for count in range(N-NX):
            tildeX=quantumgates.riggidentity(tildeX)
        return tildeX                
        
    def schwinger(c,datatype=np.complex64):
        # return a unitary operator built with the schwinger basis P^m Q^n
        # as a linear combination c(m,n)P^m U^n
        # M, N is the size of the input coefficient matrix c
        (M,N)=c.shape
        if M!=N:
            print("Error: quantumgates.schwinger only works with square matrix")
            return 0.0
        U=np.zeros([M,M],datatype)
        P=quantumgates.Xgate(N,datatype)
        Q=quantumgates.Zgate(N,datatype)
        E=np.identity(N,datatype)
        xQ=E
        for ix in range(M):
            xPQ=xQ
            for iy in range(N):
                U=U+c[ix,iy]*xPQ
                xPQ=np.matmul(P,xPQ)
            xQ=np.matmul(Q,xQ)
        return U, P, Q
                
    def isunitary(U):
        #return true if matrix U is unitary
        (M,N)=U.shape
        if M!=N:
            print("quantumgates.isunitary: matrix is not square")
            return False        
        U1=np.matrix(U)
        identity=np.matmul(U1.getH(),U1)        
        output=False
        if np.round(np.trace(identity),0)==M:
            output=True                  
        return output
    
    def Householderreflection(v,datatype=np.complex64):
        # Given a complex vector generate an Hausholder reflection
        # References Ozlov notes, Mezzadri arXiv:math-ph/0609050
        #
        # This operator H(v) is such thatn H(v).v=||v||e1
        # 
        # Version 2 november 2018, by claudio
        v=np.matrix(v)
        (N,M)=v.shape
        if M!=1:
            print("quantumgates.Householderreflection: v must be column")
            return 0.0        
        # extract firs element of v
        v1=np.asscalar(v[0])
        theta=np.asscalar(np.angle(v1))
        expitheta=np.exp(1j*theta)
        normv1=np.linalg.norm(v)
        # build unitary vector in direction 1
        e1=np.zeros([N,1],dtype=datatype)
        e1[0]=1.0
        # build u vector
        u=v+expitheta*normv1*e1
        u=u/np.linalg.norm(u)
        # build matrix
        H=np.eye(N,dtype=datatype)-2*np.matmul(u,u.getH())
        H=-np.exp(-1j*theta)*H
        return H
		
#    def beamsplitter(N,p,q,omega,phi,datatype=np.complex64):
#        # return the beam splitter matrix in the subspace of a matrix N\times NxN	
#        # for the general decomposition of a Unitary operator in beam splitters
#        # See Reck et al, PRL 73, 58 (1994
#        #
#        # Remark the index goes from 0 to N-1
##        T=np.eye(N,dtype=datatype)
##        T[p][p]=np.exp(np.1j*phi)*np.sin(omega)
##        T[p][q]=np.exp(np.1j*phi)*np.cos(omega)
##        T[q][p]=np.cos(omega)
##        T[q][q]=-np.sin(omega)
#        return T
#		
		
       
        
#%% class for useful output operations with tensorflow
class utilities:
    def printonscreen(VT):
      #print a tensor a matrix on the screen with a given precision      
      VTnp=VT.numpy()      
      N=VTnp.shape[0]
      M=VTnp.shape[1]
      for ix in range(N):
        for iy in range(M):          
          re=np.real(VTnp[ix,iy])
          im=np.imag(VTnp[ix,iy])
          print('{:+02.1f}{:+02.1f}i'.format(re,im),end=" ")
        print("") #print endlie
    def printonscreennp(VTnp):
      #print a tensor a matrix on the screen with a given precision      
      N=VTnp.shape[0]
      M=VTnp.shape[1]
      for ix in range(N):
        for iy in range(M):          
          re=np.real(VTnp[ix,iy])
          im=np.imag(VTnp[ix,iy])
          print('{:+02.1f}{:+02.1f}i'.format(re,im),end=" ")
        print("") #print endlie
        
#%% class for training quantum gates
class quantumgatesinference:
    def trainrandom(X_np,M,
                    verbose=2,
                    inputaccuracy=1e-4,
                    ntrain=100,
                    nvalid=50):
        # Given a gate with size N, generate a random unitary matrix and 
        # use a NN to train an input gate to act as the input unitary class
        #
        # Input: 
        # X_Np, gate as numpy matrix
        # M, size embedding space
        # verbose, 0 no output, 1 minimal, 2 all
        

        #%% vari import here
        ###### DA FINIRE !!!!!!!!! 
        from utilitiesquantumgates import quantumgates 
        from utilitiesquantumgates import utilities
        from tensorboardutilities import tensorboardutilities
        from datetime import datetime
        import time
        #%% datatypes
        npdatatype=np.complex64
        tfdatatype=tf.complex64
        tfrealdatatype=tf.float32 # to use double switch aboe to complex128        
        #%% number of training points
#        ntrain=100 # training set
#        nvalid=50  # validation set
        #%% epochs
        epochs=100 # maximal number of epochs
        display_steps=2 # number of steps between each validations
        #%% learning rate
        learning_rate=0.01
        #%% threshold for stopping iterations in validation cost
        threshold_valid=inputaccuracy
        #%% set the tensorboard utilities
        tensorboarddir = tensorboardutilities.getdirname();
        #%% random seed 
        timestamp = int(time.mktime(datetime.now().timetuple()))
        RANDOM_SEED=timestamp
        if verbose>1:
            print('Random seed = ' + repr(timestamp))        
        #%% define graph
        tf.compat.v1.reset_default_graph()        
        #%% summaries for tensorflow
        def variable_summaries(var):
          """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
          with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.compat.v1.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
            tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
            tf.compat.v1.summary.scalar('norm', tf.norm(var))
            tf.compat.v1.summary.histogram('histogram', var)        
        #%% seed random number generation
        tf.compat.v1.set_random_seed(RANDOM_SEED)
        np.random.seed(seed=RANDOM_SEED)                
        #%% generate the tf tensor for the input gate
        #XT=tf.constant(X_np,dtype=tfdatatype)        
        #%% unitary rigging of X
        RXT_np=quantumgates.riggunitary(X_np,M)
        RXT=tf.constant(RXT_np,dtype=tfdatatype)        
        #%% random unitary matrix
        dataU_np=quantumgates.randomU(M,npdatatype)
        U=tf.constant(dataU_np,dtype=tfdatatype)        
        #%% generate the training matrix
        W0=tf.random.uniform([M,M],dtype=tfrealdatatype)
        WC=tf.complex(tf.random_uniform([M,M],dtype=tfrealdatatype),tf.random_uniform([M,M],dtype=tfrealdatatype))
        Wreal=tf.compat.v1.get_variable("Wr",initializer=W0,dtype=tfrealdatatype)
        Wimag=tf.compat.v1.get_variable("Wi",initializer=W0,dtype=tfrealdatatype)
        W=tf.compat.v1.get_variable("W",initializer=WC,dtype=tfdatatype,trainable=False)        
        #%% transfer matrix
        transfer_matrix=tf.compat.v1.get_variable("transfer_matrix",initializer=WC,trainable=False)
        #%% place holder
        x=tf.compat.v1.placeholder(dtype=tfdatatype,shape=(M,1),name="x")     
        #%% generate training set
        xtrains=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(ntrain):
           for i in range(M):
                xtrains[i,j]=np.random.random_sample()+1j*np.random.random_sample()
        #%% normalize training set
        xtrains=tf.keras.utils.normalize(xtrains,axis=0,order=2)         
        #%% generate validation set
        xvalids=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(nvalid):
           for i in range(M):
                xvalids[i,j]=np.random.random_sample()+1j*np.random.random_sample()
        #%% normalize validation set
        xvalids=tf.keras.utils.normalize(xvalids,axis=0,order=2)             
        #%% projector that extract the first N rows from a vector M
        #project=tf.constant(quantumgates.projector(N,M,npdatatype),dtype=tfdatatype)                
        #%% equation
        with tf.name_scope("equation") as scope:
            with tf.name_scope("Wreal") as scope:
                variable_summaries(Wreal)
            with tf.name_scope("Wimag") as scope:
                variable_summaries(Wimag)
            yt=tf.matmul(RXT,x)
            W=tf.complex(Wreal,Wimag)
            transfer_matrix=tf.matmul(U,W)
            equation=tf.matmul(transfer_matrix,x)-yt
            eqreal=tf.math.real(equation)
            eqimag=tf.math.imag(equation)
            cost_function=tf.reduce_mean(tf.square(eqreal)+
                                         tf.square(eqimag))
            tf.compat.v1.summary.scalar('cost_function',cost_function)
        #%%TO DO : TRY OTHER MINIMIZER
        with tf.name_scope("train") as scope:
        #    global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        #   optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        #           cost_function, global_step=global_step)    
         #   optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        #           cost_function, global_step=global_step)
           optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
                   cost_function)
        #%% message
        if verbose>0:
            print('Running with M ' + repr(M) +
            ' ntrain ' + repr(ntrain) +
            ' nvalid ' + repr(nvalid))                
        #%% writer
        train_writer=tf.compat.v1.summary.FileWriter(tensorboarddir)
        merged=tf.compat.v1.summary.merge_all()
           
        #%%
        xtmp=np.zeros((M,1),dtype=npdatatype)
        with tf.compat.v1.Session()  as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            Tinitial=transfer_matrix.eval()
            for epoch in range(epochs):
                avg_cost=0.
                for i in range(ntrain):
                   xtmp=np.reshape(xtrains[0:M,i],(M,1))
                   sess.run(optimizer,feed_dict={x: xtmp})
                   avg_cost+=sess.run(cost_function, feed_dict={x: xtmp})
                   summary=sess.run(merged, feed_dict={x: xtmp}) 
                   train_writer.add_summary(summary,i+epoch*epochs)
                avg_cost=avg_cost/ntrain
                # messagers
                if epoch % display_steps == 0:
                   # evaluate the validation error
                   avg_cost_valid=0.
                   for i in range(nvalid):
                       xtmp_valid=np.reshape(xvalids[0:M,i],(M,1))
                       avg_cost_valid+=sess.run(cost_function, feed_dict=
                                                {x: xtmp_valid})
                   avg_cost_valid=avg_cost_valid/nvalid
                   if verbose>1:
                       print('epoch '+repr(epoch))
                       print('cost '+repr(avg_cost))
                       print('valid cost '+repr(avg_cost_valid))
                 # check the validation cost and if needed exit the iteration
                if avg_cost_valid < threshold_valid:         
                    if verbose:
                        print('Convergence in validation reached at epoch ' 
                              + repr(epoch))
                    break
                if epoch>=epochs-1:
                    if verbose>0:
                        print('No convergence, maximal epochs reached '
                              +repr(epochs))            
            Tfinal=transfer_matrix.eval()
            Wfinal=W.eval()
            TVV=tf.matmul(W,W,adjoint_a=True).eval()
        #    print('Determinant Structure matrix ' + repr(np.linalg.det(dataU_np)))
        #%%
        if verbose>1:
            print("Final Sinput=W")
            utilities.printonscreennp(Wfinal)    
            print("Final TV V for unitarity ")
            utilities.printonscreennp(TVV)    
            print("Initial T")
            utilities.printonscreennp(Tinitial)    
            print("Final T")
            utilities.printonscreennp(Tfinal)    
                
        #%%
        sess.close()

        
        #%% set the output dictionary of parameters
        out=dict();
        out['accuracy']=threshold_valid
        out['epoch']=epoch
        out['ntrain']=ntrain
        out['nvalid']=nvalid
        out['N']=X_np.shape[0]
        out['M']=M
        out['X']=X_np
		
        
        
        return out, Wfinal, Tfinal, Tinitial

#%%%%        
    def traincomplex(X_np,U_np,
                    verbose=2,
                    inputaccuracy=1e-4,
                    ntrain=100,
                    nvalid=50):
        # Given a gate with size N, and a complex system described by an input MxM U_np transfer matrix
        # use a NN to train an input gate to act as the input unitary class
        #
        # The input gate is only a phase gate, described by a diagonal matrix
        # with diagonal exp(i phi1), exp(i phi2), ..., exp(i phin)
        #
        # with phi1, phi2, ..., phin are trainable
        #
        # TO DO, make batch training (not use it can train without batch)
        #
        # Date: 5 April 2019, by Claudio
        #
        # Input: 
        # X_np, gate as numpy matrix
		# U_np, complex system unitary matrix (not checked if unitary) a numpy matrix
        # verbose, 0 no output, 1 minimal, 2 all
        

        #%% vari import here
        ###### DA FINIRE !!!!!!!!! 
        from utilitiesquantumgates import quantumgates 
        from utilitiesquantumgates import utilities
        from tensorboardutilities import tensorboardutilities
        from datetime import datetime
        import time
        #%% datatypes
        npdatatype=np.complex64
        tfdatatype=tf.complex64
        tfrealdatatype=tf.float32 # to use double switch aboe to complex128        
        #%% number of training points
		#        ntrain=100 # training set
		#        nvalid=50  # validation set
        #%% epochs
        epochs=100 # maximal number of epochs
        display_steps=2 # number of steps between each validations
        #%% learning rate
        learning_rate=0.01
        #%% threshold for stopping iterations in validation cost
        threshold_valid=inputaccuracy
        #%% set the tensorboard utilities
        tensorboarddir = tensorboardutilities.getdirname();
        #%% random seed 
        timestamp = int(time.mktime(datetime.now().timetuple()))
        RANDOM_SEED=timestamp
        if verbose>1:
            print('Random seed = ' + repr(timestamp))        
        #%% define graph
        tf.reset_default_graph()        
        #%% summaries for tensorflow
        def variable_summaries(var):
          """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
          with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.scalar('norm', tf.norm(var))
            tf.summary.histogram('histogram', var)        
        #%% seed random number generation
        tf.set_random_seed(RANDOM_SEED)
        np.random.seed(seed=RANDOM_SEED)                
        #%% generate the tf tensor for the input gate
        #XT=tf.constant(X_np,dtype=tfdatatype)        
		#Extract N and M in input
        N=X_np.shape[0]
        M=U_np.shape[0]
        #%% unitary rigging of X
        RXT_np=quantumgates.riggunitary(X_np,M)
        RXT=tf.constant(RXT_np,dtype=tfdatatype)        
        #%% random unitary matrix
        U=tf.constant(U_np,dtype=tfdatatype)        
        #%% generate the training matrix
        W0=tf.random_uniform([M,M],dtype=tfrealdatatype)
        WC=tf.complex(tf.random_uniform([M,M],dtype=tfrealdatatype),tf.random_uniform([M,M],dtype=tfrealdatatype))
        Wreal=tf.get_variable("Wr",initializer=W0,dtype=tfrealdatatype)
        Wimag=tf.get_variable("Wi",initializer=W0,dtype=tfrealdatatype)
        W=tf.get_variable("W",initializer=WC,dtype=tfdatatype,trainable=False)        
        #%% transfer matrix
        transfer_matrix=tf.get_variable("transfer_matrix",initializer=WC,trainable=False)
        #%% place holder
        x=tf.placeholder(dtype=tfdatatype,shape=(M,1),name="x")        
        #%% generate training set
        xtrains=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(ntrain):
           for i in range(M):
                xtrains[i,j]=np.random.random_sample()+1j*np.random.random_sample()
        #%% normalize training set
        xtrains=tf.keras.utils.normalize(xtrains,axis=0,order=2)         
        #%% generate validation set
        xvalids=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(nvalid):
           for i in range(M):
                xvalids[i,j]=np.random.random_sample()+1j*np.random.random_sample()
        #%% normalize validation set
        xvalids=tf.keras.utils.normalize(xvalids,axis=0,order=2)             
        #%% projector that extract the first N rows from a vector M
        #project=tf.constant(quantumgates.projector(N,M,npdatatype),dtype=tfdatatype)                
        #%% equation
        with tf.name_scope("equation") as scope:
            with tf.name_scope("Wreal") as scope:
                variable_summaries(Wreal)
            with tf.name_scope("Wimag") as scope:
                variable_summaries(Wimag)
            yt=tf.matmul(RXT,x)
            W=tf.complex(Wreal,Wimag)
            transfer_matrix=tf.matmul(U,W)
            equation=tf.matmul(transfer_matrix,x)-yt
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
        #%% message
        if verbose>0:
            print('Running with M ' + repr(M) +
            ' ntrain ' + repr(ntrain) +
            ' nvalid ' + repr(nvalid))                
        #%% writer
        train_writer=tf.summary.FileWriter(tensorboarddir)
        merged=tf.summary.merge_all()
           
        #%%
        xtmp=np.zeros((M,1),dtype=npdatatype)
        with tf.Session()  as sess:
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            Tinitial=transfer_matrix.eval()
            for epoch in range(epochs):
                avg_cost=0.
                for i in range(ntrain):
                   xtmp=np.reshape(xtrains[0:M,i],(M,1))
                   sess.run(optimizer,feed_dict={x: xtmp})
                   avg_cost+=sess.run(cost_function, feed_dict={x: xtmp})
                   summary=sess.run(merged, feed_dict={x: xtmp}) 
                   train_writer.add_summary(summary,i+epoch*epochs)
                avg_cost=avg_cost/ntrain
                # messagers
                if epoch % display_steps == 0:
                   # evaluate the validation error
                   avg_cost_valid=0.
                   for i in range(nvalid):
                       xtmp_valid=np.reshape(xvalids[0:M,i],(M,1))
                       avg_cost_valid+=sess.run(cost_function, feed_dict=
                                                {x: xtmp_valid})
                   avg_cost_valid=avg_cost_valid/nvalid
                   if verbose>1:
                       print('epoch '+repr(epoch))
                       print('cost '+repr(avg_cost))
                       print('valid cost '+repr(avg_cost_valid))
                 # check the validation cost and if needed exit the iteration
                if avg_cost_valid < threshold_valid:         
                    if verbose:
                        print('Convergence in validation reached at epoch ' 
                              + repr(epoch))
                    break
                if epoch>=epochs-1:
                    if verbose>0:
                        print('No convergence, maximal epochs reached '
                              +repr(epochs))            
            Tfinal=transfer_matrix.eval()
            Wfinal=W.eval()
            TVV=tf.matmul(W,W,adjoint_a=True).eval()
        #    print('Determinant Structure matrix ' + repr(np.linalg.det(dataU_np)))
        #%%
        if verbose>1:
            print("Final Sinput=W")
            utilities.printonscreennp(Wfinal)    
            print("Final TV V for unitarity ")
            utilities.printonscreennp(TVV)    
            print("Initial T")
            utilities.printonscreennp(Tinitial)    
            print("Final T")
            utilities.printonscreennp(Tfinal)    
                
        #%%
        sess.close()

        
        #%% set the output dictionary of parameters
        out=dict();
        out['accuracy']=threshold_valid
        out['epoch']=epoch
        out['ntrain']=ntrain
        out['nvalid']=nvalid
        out['N']=N
        out['M']=M
        out['X']=X_np
        out['U']=U_np
		
        
        
        return out, Wfinal, Tfinal, Tinitial
        
#%% class for training SLM with single input
class SLM:
    def trainSLMsingleinputquantized(X_np,U_np,
                     verbose=2,
                    inputaccuracy=1e-4,
                    epochs=10,display_steps=100,
                    realMIN=-1.0, realMAX=1.0,
                    imagMIN=0.0, imagMAX=0.0,
                    quantizedbits=8):
        # Given a gate with size N, generate a random unitary matrix and 
        # use a NN to train an input gate to act as the input unitary class
        #
        # Input: 
        # X_Np, gate as numpy matrix
        # M, size embedding space
        # verbose, 0 no output, 1 minimal, 2 steps, 3 all
        #
        # Use single input SLM
        #
        # WrealMAX, WrealMIN, maximal and minimal value for Wreal
        #
        # WimagMAX, WimagMIN, maximal and minimal value for Wimag (if both 0 is a real weigth)
        #
        # quantized bits
        

        #%% vari import here
        ###### DA FINIRE !!!!!!!!! 
        from utilitiesquantumgates import quantumgates 
        from utilitiesquantumgates import utilities
        from tensorboardutilities import tensorboardutilities
        from datetime import datetime
        import time
        #%% datatypes
        npdatatype=np.complex64
        tfdatatype=tf.complex64
        tfrealdatatype=tf.float32 # to use double switch aboe to complex128        
        #%% number of training points
        ntrain=1
        nvalid=1
        #%% learning rate
        learning_rate=0.01
        #%% threshold for stopping iterations in validation cost
        threshold_valid=inputaccuracy
        #%% set the tensorboard utilities
        tensorboarddir = tensorboardutilities.getdirname();
        #%% random seed 
        timestamp = int(time.mktime(datetime.now().timetuple()))
        RANDOM_SEED=timestamp
        if verbose>1:
            print('Random seed = ' + repr(timestamp))        
        #%% define graph
        tf.compat.v1.reset_default_graph()        
        #%% summaries for tensorflow
        def variable_summaries(var):
          """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
          with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.compat.v1.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
            tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
            tf.compat.v1.summary.scalar('norm', tf.norm(var))
            tf.compat.v1.summary.histogram('histogram', var)        
        #%% seed random number generation
        tf.compat.v1.set_random_seed(RANDOM_SEED)
        np.random.seed(seed=RANDOM_SEED)                
        #%% Extract N and M in input
        N=X_np.shape[0]
        M=U_np.shape[0]
        if M<N:
            print("Error: embedding dimension M cannot be smaller then N")
            return
        #%% unitary rigging of X
        RXT_np=quantumgates.riggunitary(X_np,M)
        RXT=tf.constant(RXT_np,dtype=tfdatatype)
        print(RXT)        
        #%% unitary rigging of X
#        XT=tf.constant(X_np)
        #%% random unitary matrix
        U=tf.constant(U_np,dtype=tfdatatype)        
        #%% generate the training matrix
        W0=tf.random.uniform([M,M],dtype=tfrealdatatype)
        WC=tf.complex(tf.random.uniform([M,M],dtype=tfrealdatatype),tf.random.uniform([M,M],dtype=tfrealdatatype))
        Wreal=tf.compat.v1.get_variable("Wr",initializer=W0,dtype=tfrealdatatype)
        Wimag=tf.compat.v1.get_variable("Wi",initializer=W0,dtype=tfrealdatatype)
        W=tf.compat.v1.get_variable("W",initializer=WC,dtype=tfdatatype,trainable=False)        
        #%% transfer matrix
        transfer_matrix=tf.compat.v1.get_variable("transfer_matrix",initializer=WC,trainable=False)
        #%% place holder
        x=tf.compat.v1.placeholder(dtype=tfdatatype,shape=(M,1),name="x")     
        #%% generate training set
        xtrains=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(ntrain):
           for i in range(M):
                xtrains[i,j]=1.0
        #%% normalize training set
        xtrains=tf.keras.utils.normalize(xtrains,axis=0,order=2)         
        #%% generate validation set
        xvalids=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(nvalid):
           for i in range(M):
                xvalids[i,j]=1.0
        #%% normalize validation set
        xvalids=tf.keras.utils.normalize(xvalids,axis=0,order=2)             
        #%% projector that extract the first N rows from a vector M
        #project=tf.constant(quantumgates.projector(N,M,npdatatype),dtype=tfdatatype)                
        #%% equation
        with tf.name_scope("equation") as scope:
            with tf.name_scope("Wreal") as scope:
                variable_summaries(Wreal)
            with tf.name_scope("Wimag") as scope:
                variable_summaries(Wimag)
            yt=tf.matmul(RXT,x)
            #clipping the weigths
            Wreal=tf.clip_by_value(Wreal,realMIN,realMAX)
            Wimag=tf.clip_by_value(Wimag,imagMIN,imagMAX)            
            # quantize
            Wreal=tf.quantization.quantize_and_dequantize(Wreal,realMIN,realMAX,signed_input=False,num_bits=quantizedbits)
            Wimag=tf.quantization.quantize_and_dequantize(Wimag,imagMIN,imagMAX,signed_input=False,num_bits=quantizedbits)
            # build the matrices (phase only modulator)
            #W=tf.complex(cWreal,cWimag)
            W=tf.complex(tf.cos(Wreal),tf.sin(Wreal))
            transfer_matrix=tf.matmul(U,W)
            equation=tf.matmul(transfer_matrix,x)-yt
            eqreal=tf.math.real(equation)
            eqimag=tf.math.imag(equation)
            cost_function=tf.reduce_mean(tf.square(eqreal)+
                                         tf.square(eqimag))
            tf.compat.v1.summary.scalar('cost_function',cost_function)
        #%%TO DO : TRY OTHER MINIMIZER
        with tf.name_scope("train") as scope:
        #    global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        #   optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        #           cost_function, global_step=global_step)    
         #   optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        #           cost_function, global_step=global_step)
           optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
                   cost_function)
        #%% message
        if verbose>0:
            print('Running with M ' + repr(M) +
            ' N ' + repr(N) +
            ' ntrain ' + repr(ntrain) +
            ' nvalid ' + repr(nvalid))                
        #%% writer
        train_writer=tf.compat.v1.summary.FileWriter(tensorboarddir)
        merged=tf.compat.v1.summary.merge_all()
           
        #%%
        xtmp=np.zeros((M,1),dtype=npdatatype)
        with tf.compat.v1.Session()  as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            Tinitial=transfer_matrix.eval()
            for epoch in range(epochs):
                avg_cost=0.
                for i in range(ntrain):
                   xtmp=np.reshape(xtrains[0:M,i],(M,1))
                   sess.run(optimizer,feed_dict={x: xtmp})
                   avg_cost+=sess.run(cost_function, feed_dict={x: xtmp})
                   summary=sess.run(merged, feed_dict={x: xtmp}) 
                   train_writer.add_summary(summary,i+epoch*epochs)
                avg_cost=avg_cost/ntrain
                # messagers
                if epoch % display_steps == 0:
                   # evaluate the validation error
                   avg_cost_valid=0.
                   for i in range(nvalid):
                       xtmp_valid=np.reshape(xvalids[0:M,i],(M,1))
                       avg_cost_valid+=sess.run(cost_function, feed_dict=
                                                {x: xtmp_valid})
                   avg_cost_valid=avg_cost_valid/nvalid
                   if verbose>1:
                       print('epoch '+repr(epoch))
                       print('cost '+repr(avg_cost))
                       print('valid cost '+repr(avg_cost_valid))
                 # check the validation cost and if needed exit the iteration
                if avg_cost_valid < threshold_valid:         
                    if verbose:
                        print('Convergence in validation reached at epoch ' 
                              + repr(epoch))
                    break
                if epoch>=epochs-1:
                    if verbose>0:
                        print('No convergence, maximal epochs reached '
                              +repr(epochs))            
            Tfinal=transfer_matrix.eval()
            rWfinal=Wreal.eval()
            iWfinal=Wimag.eval()
            Wfinal=W.eval()
            TVV=tf.matmul(W,W,adjoint_a=True).eval()
        #    print('Determinant Structure matrix ' + repr(np.linalg.det(dataU_np)))
        #%%
        if verbose>2:
            print("Final Wreal")
            utilities.printonscreennp(rWfinal)    
            print("Final Wimag")
            utilities.printonscreennp(iWfinal)    
            print("Final Sinput=W")
            utilities.printonscreennp(Wfinal)    
            print("Final TV V for unitarity ")
            utilities.printonscreennp(TVV)    
            print("Initial T")
            utilities.printonscreennp(Tinitial)    
            print("Final T")
            utilities.printonscreennp(Tfinal)    
                
        #%%
        sess.close()

        
        #%% set the output dictionary of parameters
        out=dict();
        out['accuracy']=threshold_valid
        out['epoch']=epoch
        out['ntrain']=ntrain
        out['nvalid']=nvalid
        out['N']=X_np.shape[0]
        out['M']=M
        out['X']=X_np
		
        
        
        return out, Wfinal, Tfinal, Tinitial

    def complexqtzd(X_np,U_np,
                    verbose=2,
                    inputaccuracy=1e-4,
                    epochs=10,display_steps=100,
                    realMIN=-1.0, realMAX=1.0,
                    imagMIN=0.0, imagMAX=0.0,
                    nbits=8):
        #%% Train a single input SLM with complex matrix
        # Given a gate with size N, generate a random unitary matrix and 
        # use a NN to train an input gate to act as the input unitary class
        #
        # Input: 
        # X_np, gate as numpy matrix
        # U_np, unitary matrix for medium
        # verbose, 0 no output, 1 minimal, 2 steps, 3 all
        #
        # Use single input SLM with complex matrix
        #
        # WrealMAX, WrealMIN, maximal and minimal value for Wreal
        #
        # WimagMAX, WimagMIN, maximal and minimal value for Wimag 
        # If WimagMAX=WimagMIN=0 is a amplitude modulator 
        
        
        #%%
        from utilitiesquantumgates import quantumgates 
        from utilitiesquantumgates import utilities
        from tensorboardutilities import tensorboardutilities
        from datetime import datetime
        import time
        # datatypes
        npdatatype=np.complex64
        tfdatatype=tf.complex64
        tfrealdatatype=tf.float32 # to use double switch aboe to complex128        
        #%% number of training points
        ntrain=1
        nvalid=1
        #%% learning rate
        learning_rate=0.01
        #%% threshold for stopping iterations in validation cost
        threshold_valid=inputaccuracy
        #%% set the tensorboard utilities
        tensorboarddir = tensorboardutilities.getdirname();
        #%% random seed 
        timestamp = int(time.mktime(datetime.now().timetuple()))
        RANDOM_SEED=timestamp
        if verbose>1:
            print('Random seed = ' + repr(timestamp))        
        #%% define graph
        tf.compat.v1.reset_default_graph()        
        #%% summaries for tensorflow
        def variable_summaries(var):
          """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
          with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.scalar('norm', tf.norm(var))
            tf.summary.histogram('histogram', var)        
        #%% seed random number generation
        tf.compat.v1.set_random_seed(RANDOM_SEED)
        np.random.seed(seed=RANDOM_SEED)                
        #%% Extract N and M in input
        N=X_np.shape[0]
        M=U_np.shape[0]
        if M<N:
            print("Error: embedding dimension M cannot be smaller then N")
            return
        #%% unitary rigging of X
        RXT_np=quantumgates.riggunitary(X_np,M)
        RXT=tf.constant(RXT_np,dtype=tfdatatype)
        #%% random unitary matrix
        U=tf.constant(U_np,dtype=tfdatatype)        
        #%% generate the training matrix
        W0=tf.random.uniform([M,M],dtype=tfrealdatatype)
        WC=tf.complex(tf.random.uniform([M,M],dtype=tfrealdatatype),tf.random.uniform([M,M],dtype=tfrealdatatype))
        Wreal=tf.compat.v1.get_variable("Wr",initializer=W0,dtype=tfrealdatatype)
        Wimag=tf.compat.v1.get_variable("Wi",initializer=W0,dtype=tfrealdatatype)
        W=tf.compat.v1.get_variable("W",initializer=WC,dtype=tfdatatype,trainable=False)        
        #%% transfer matrix
        transfer_matrix=tf.compat.v1.get_variable("transfer_matrix",initializer=WC,trainable=False)
        #%% current output
        yC=tf.complex(tf.random.uniform([M,1],dtype=tfrealdatatype),tf.random.uniform([M,1],dtype=tfrealdatatype))
        yout=tf.compat.v1.get_variable("current_y",initializer=yC,trainable=False)
        yt=tf.compat.v1.get_variable("target_y",initializer=yC,trainable=False)
        #%% place holder
        x=tf.compat.v1.placeholder(dtype=tfdatatype,shape=(M,1),name="x")     
        #%% generate training set, one single input all 1 to N, M-N zeros
        xtrains=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(ntrain):
           for i in range(N):
                xtrains[i,j]=1.0
        #%% normalize training set
        #xtrains=tf.keras.utils.normalize(xtrains,axis=0,order=2)         
        #%% generate validation set (here equal to the training)
        xvalids=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(nvalid):
           for i in range(N):
                xvalids[i,j]=1.0
        #%% normalize validation set
        #xvalids=tf.keras.utils.normalize(xvalids,axis=0,order=2)             
        #%% equation
        with tf.name_scope("equation") as scope:
            with tf.name_scope("Wreal") as scope:
                variable_summaries(Wreal)
            with tf.name_scope("Wimag") as scope:
                variable_summaries(Wimag)
            yt=tf.matmul(RXT,x)
            #clipping the weigths
            Wreal=tf.clip_by_value(Wreal,realMIN,realMAX)
            Wimag=tf.clip_by_value(Wimag,imagMIN,imagMAX)            
            # quantize
            Wreal=tf.quantization.quantize_and_dequantize(Wreal,realMIN,realMAX,signed_input=False,num_bits=nbits)
            Wimag=tf.quantization.quantize_and_dequantize(Wimag,imagMIN,imagMAX,signed_input=False,num_bits=nbits)
            # build the matrices (phase only modulator)
            W=tf.complex(Wreal,Wimag)
            transfer_matrix=tf.matmul(U,W)
            yout=tf.matmul(transfer_matrix,x)
            equation=yout-yt
            eqreal=tf.math.real(equation)
            eqimag=tf.math.imag(equation)
            cost_function=tf.reduce_mean(tf.square(eqreal)+
                                         tf.square(eqimag))
            tf.compat.v1.summary.scalar('cost_function',cost_function)
        #%%TO DO : TRY OTHER MINIMIZER
        with tf.name_scope("train") as scope:
        #    global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        #   optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        #           cost_function, global_step=global_step)    
         #   optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        #           cost_function, global_step=global_step)
           optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
                   cost_function)
        #%% message
        if verbose>0:
            print('Running with M ' + repr(M) +
            ' N ' + repr(N) +
            ' ntrain ' + repr(ntrain) +
            ' nvalid ' + repr(nvalid))                
        #%% writer
        train_writer=tf.compat.v1.summary.FileWriter(tensorboarddir)
        merged=tf.compat.v1.summary.merge_all()
           
        #%%
        xtmp=np.zeros((M,1),dtype=npdatatype)
        with tf.compat.v1.Session()  as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            Tinitial=transfer_matrix.eval()
            for epoch in range(epochs):
                avg_cost=0.
                for i in range(ntrain):
                   xtmp=np.reshape(xtrains[0:M,i],(M,1))
                   sess.run(optimizer,feed_dict={x: xtmp})
                   avg_cost+=sess.run(cost_function, feed_dict={x: xtmp})
                   summary=sess.run(merged, feed_dict={x: xtmp}) 
                   train_writer.add_summary(summary,i+epoch*epochs)
                avg_cost=avg_cost/ntrain
                # messagers
                if epoch % display_steps == 0:
                   # evaluate the validation error
                   avg_cost_valid=0.
                   for i in range(nvalid):
                       xtmp_valid=np.reshape(xvalids[0:M,i],(M,1))
                       avg_cost_valid+=sess.run(cost_function, feed_dict=
                                                {x: xtmp_valid})
                   avg_cost_valid=avg_cost_valid/nvalid
                   if verbose>1:
                       print('epoch '+repr(epoch))
                       print('cost '+repr(avg_cost))
                       print('valid cost '+repr(avg_cost_valid))
                 # check the validation cost and if needed exit the iteration
                if avg_cost_valid < threshold_valid:         
                    if verbose:
                        print('Convergence in validation reached at epoch ' 
                              + repr(epoch))
                    break
                if epoch>=epochs-1:
                    if verbose>0:
                        print('No convergence, maximal epochs reached '
                              +repr(epochs))            
            Tfinal=transfer_matrix.eval()
            ytargetf=sess.run(yt, feed_dict={x: xtmp}) 
            youtf=sess.run(yout, feed_dict={x: xtmp}) 
            rWfinal=Wreal.eval()
            iWfinal=Wimag.eval()
            Wfinal=W.eval()
            TVV=tf.matmul(W,W,adjoint_a=True).eval()
        #    print('Determinant Structure matrix ' + repr(np.linalg.det(dataU_np)))
        #%%
        if verbose>2:
            print("Final Wreal")
            utilities.printonscreennp(rWfinal)    
            print("Final Wimag")
            utilities.printonscreennp(iWfinal)    
            print("Final Sinput=W")
            utilities.printonscreennp(Wfinal)    
            print("Final TV V for unitarity ")
            utilities.printonscreennp(TVV)    
            print("Initial T")
            utilities.printonscreennp(Tinitial)    
            print("Final T")
            utilities.printonscreennp(Tfinal)    
                
        #%%
        sess.close()

        
        #%% set the output dictionary of parameters
        out=dict();
        out['accuracy']=threshold_valid
        out['epoch']=epoch
        out['ntrain']=ntrain
        out['nvalid']=nvalid
        out['N']=X_np.shape[0]
        out['M']=M
        out['X']=X_np
        out['xtrain']=xtrains
        out['Wreal']=rWfinal
        out['Wimag']=iWfinal
        out['Wfinal']=Wfinal
        out['Tfinal']=Tfinal
        out['yt']=ytargetf
        out['y']=youtf
        out['cost']=avg_cost_valid
		
        
        
        return out
    
    def complex(X_np,U_np,
                    verbose=2,
                    inputaccuracy=1e-4,
                    epochs=10,display_steps=100,
                    realMIN=-1.0, realMAX=1.0,
                    imagMIN=0.0, imagMAX=0.0):
        #%% Train a single input SLM with complex matrix
        # Given a gate with size N, generate a random unitary matrix and 
        # use a NN to train an input gate to act as the input unitary class
        #
        # Input: 
        # X_np, gate as numpy matrix
        # U_np, unitary matrix for medium
        # verbose, 0 no output, 1 minimal, 2 steps, 3 all
        #
        # Use single input SLM with complex matrix
        #
        # WrealMAX, WrealMIN, maximal and minimal value for Wreal
        #
        # WimagMAX, WimagMIN, maximal and minimal value for Wimag 
        # If WimagMAX=WimagMIN=0 is a amplitude modulator 
        
        
        #%%
        from utilitiesquantumgates import quantumgates 
        from utilitiesquantumgates import utilities
        from tensorboardutilities import tensorboardutilities
        from datetime import datetime
        import time
        # datatypes
        npdatatype=np.complex64
        tfdatatype=tf.complex64
        tfrealdatatype=tf.float32 # to use double switch aboe to complex128        
        #%% number of training points
        ntrain=1
        nvalid=1
        #%% learning rate
        learning_rate=0.01
        #%% threshold for stopping iterations in validation cost
        threshold_valid=inputaccuracy
        #%% set the tensorboard utilities
        tensorboarddir = tensorboardutilities.getdirname();
        #%% random seed 
        timestamp = int(time.mktime(datetime.now().timetuple()))
        RANDOM_SEED=timestamp
        if verbose>1:
            print('Random seed = ' + repr(timestamp))        
        #%% define graph
        tf.compat.v1.reset_default_graph()        
        #%% summaries for tensorflow
        def variable_summaries(var):
          """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
          with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.scalar('norm', tf.norm(var))
            tf.summary.histogram('histogram', var)        
        #%% seed random number generation
        tf.compat.v1.set_random_seed(RANDOM_SEED)
        np.random.seed(seed=RANDOM_SEED)                
        #%% Extract N and M in input
        N=X_np.shape[0]
        M=U_np.shape[0]
        if M<N:
            print("Error: embedding dimension M cannot be smaller then N")
            return
        #%% unitary rigging of X
        RXT_np=quantumgates.riggunitary(X_np,M)
        RXT=tf.constant(RXT_np,dtype=tfdatatype)
        #%% random unitary matrix
        U=tf.constant(U_np,dtype=tfdatatype)        
        #%% generate the training matrix
        W0=tf.random_uniform([M,M],dtype=tfrealdatatype)
        WC=tf.complex(tf.random.uniform([M,M],dtype=tfrealdatatype),tf.random.uniform([M,M],dtype=tfrealdatatype))
        Wreal=tf.compat.v1.get_variable("Wr",initializer=W0,dtype=tfrealdatatype)
        Wimag=tf.compat.v1.get_variable("Wi",initializer=W0,dtype=tfrealdatatype)
        W=tf.get_variable("W",initializer=WC,dtype=tfdatatype,trainable=False)        
        #%% transfer matrix
        transfer_matrix=tf.get_variable("transfer_matrix",initializer=WC,trainable=False)
        #%% current output
        yC=tf.complex(tf.random.uniform([M,1],dtype=tfrealdatatype),tf.random.uniform([M,1],dtype=tfrealdatatype))
        yout=tf.get_variable("current_y",initializer=yC,trainable=False)
        yt=tf.get_variable("target_y",initializer=yC,trainable=False)
        #%% place holder
        x=tf.placeholder(dtype=tfdatatype,shape=(M,1),name="x")     
        #%% generate training set, one single input all 1 to N, M-N zeros
        xtrains=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(ntrain):
           for i in range(N):
                xtrains[i,j]=1.0
        #%% normalize training set
        #xtrains=tf.keras.utils.normalize(xtrains,axis=0,order=2)         
        #%% generate validation set (here equal to the training)
        xvalids=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(nvalid):
           for i in range(N):
                xvalids[i,j]=xtrains[i,j]
        #%% normalize validation set
        #xvalids=tf.keras.utils.normalize(xvalids,axis=0,order=2)             
        #%% equation
        with tf.name_scope("equation") as scope:
            with tf.name_scope("Wreal") as scope:
                variable_summaries(Wreal)
            with tf.name_scope("Wimag") as scope:
                variable_summaries(Wimag)
            yt=tf.matmul(RXT,x)
            #clipping the weigths
            Wreal=tf.clip_by_value(Wreal,realMIN,realMAX)
            Wimag=tf.clip_by_value(Wimag,imagMIN,imagMAX)            
            # build the matrices (phase only modulator)
            W=tf.complex(Wreal,Wimag)
            transfer_matrix=tf.matmul(U,W)
            yout=tf.matmul(transfer_matrix,x)
            equation=yout-yt
            eqreal=tf.math.real(equation)
            eqimag=tf.math.imag(equation)
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
           optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
                   cost_function)
        #%% message
        if verbose>0:
            print('Running with M ' + repr(M) +
            ' N ' + repr(N) +
            ' ntrain ' + repr(ntrain) +
            ' nvalid ' + repr(nvalid))                
        #%% writer
        train_writer=tf.summary.FileWriter(tensorboarddir)
        merged=tf.summary.merge_all()
           
        #%%
        xtmp=np.zeros((M,1),dtype=npdatatype)
        with tf.Session()  as sess:
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            Tinitial=transfer_matrix.eval()
            for epoch in range(epochs):
                avg_cost=0.
                for i in range(ntrain):
                   xtmp=np.reshape(xtrains[0:M,i],(M,1))
                   sess.run(optimizer,feed_dict={x: xtmp})
                   avg_cost+=sess.run(cost_function, feed_dict={x: xtmp})
                   summary=sess.run(merged, feed_dict={x: xtmp}) 
                   train_writer.add_summary(summary,i+epoch*epochs)
                avg_cost=avg_cost/ntrain
                # messagers
                if epoch % display_steps == 0:
                   # evaluate the validation error
                   avg_cost_valid=0.
                   for i in range(nvalid):
                       xtmp_valid=np.reshape(xvalids[0:M,i],(M,1))
                       avg_cost_valid+=sess.run(cost_function, feed_dict=
                                                {x: xtmp_valid})
                   avg_cost_valid=avg_cost_valid/nvalid
                   if verbose>1:
                       print('epoch '+repr(epoch))
                       print('cost '+repr(avg_cost))
                       print('valid cost '+repr(avg_cost_valid))
                 # check the validation cost and if needed exit the iteration
                if avg_cost_valid < threshold_valid:         
                    if verbose:
                        print('Convergence in validation reached at epoch ' 
                              + repr(epoch))
                    break
                if epoch>=epochs-1:
                    if verbose>0:
                        print('No convergence, maximal epochs reached '
                              +repr(epochs))            
            Tfinal=transfer_matrix.eval()
            ytargetf=sess.run(yt, feed_dict={x: xtmp}) 
            youtf=sess.run(yout, feed_dict={x: xtmp}) 
            rWfinal=Wreal.eval()
            iWfinal=Wimag.eval()
            Wfinal=W.eval()
            TVV=tf.matmul(W,W,adjoint_a=True).eval()
        #    print('Determinant Structure matrix ' + repr(np.linalg.det(dataU_np)))
        #%%
        if verbose>2:
            print("Final Wreal")
            utilities.printonscreennp(rWfinal)    
            print("Final Wimag")
            utilities.printonscreennp(iWfinal)    
            print("Final Sinput=W")
            utilities.printonscreennp(Wfinal)    
            print("Final TV V for unitarity ")
            utilities.printonscreennp(TVV)    
            print("Initial T")
            utilities.printonscreennp(Tinitial)    
            print("Final T")
            utilities.printonscreennp(Tfinal)    
                
        #%%
        sess.close()

        
        #%% set the output dictionary of parameters
        out=dict();
        out['accuracy']=threshold_valid
        out['epoch']=epoch
        out['ntrain']=ntrain
        out['nvalid']=nvalid
        out['N']=X_np.shape[0]
        out['M']=M
        out['X']=X_np
        out['xtrain']=xtrains
        out['Wreal']=rWfinal
        out['Wimag']=iWfinal
        out['Wfinal']=Wfinal
        out['Tfinal']=Tfinal
        out['yt']=ytargetf
        out['y']=youtf
        out['cost']=avg_cost_valid
		
        
        
        return out

    def phaseonly(X_np,U_np,
                    verbose=2,
                    inputaccuracy=1e-4,
                    epochs=10,
                    display_steps=100,
                     ntrain=1,
                     nvalid=1):
        #%% Train a single input SLM with complex matrix
        # Given a gate with size N, generate a random unitary matrix and 
        # use a NN to train an input gate to act as the input unitary class
        #
        # Input: 
        # X_np, gate as numpy matrix
        # U_np, unitary matrix for medium
        # verbose, 0 no output, 1 minimal, 2 steps, 3 all
        #
        # Use single input SLM with complex matrix
        #
        # WrealMAX, WrealMIN, maximal and minimal value for Wreal, here are set in the range -pi, pi to mimic a phase only modulator
        #
        # WimagMAX, WimagMIN, maximal and minimal value for Wimag (if both 0 is a real weigth)
        #
        # NB: Use a single training and validation as it represent a phase only modulator with input a plane wave
        #
        #%%
        from utilitiesquantumgates import quantumgates 
        from utilitiesquantumgates import utilities
        from tensorboardutilities import tensorboardutilities
        from datetime import datetime
        import time
        import math 
        # datatypes
        npdatatype=np.complex64
        tfdatatype=tf.complex64
        tfrealdatatype=tf.float32 # to use double switch aboe to complex128        
        #%% learning rate
        learning_rate = 0.01
        #%% threshold for stopping iterations in validation cost
        threshold_valid = inputaccuracy
        #%% set the tensorboard utilities
        tensorboarddir = tensorboardutilities.getdirname();
        #%% bounds for weigths
        realMAX = math.pi
        realMIN = -math.pi
        imagMAX = 0.0
        imagMIN = 0.0        
        #%% random seed 
        timestamp = int(time.mktime(datetime.now().timetuple()))
        RANDOM_SEED = timestamp
        #%% define graph (compatibility for tf 2)
        tf.compat.v1.reset_default_graph()        
        #%% summaries for tensorflow
        def variable_summaries(var):
          """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
          with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.compat.v1.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
            tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
            tf.compat.v1.summary.scalar('norm', tf.norm(var))
            tf.compat.v1.summary.histogram('histogram', var)        
        #%% seed random number generation
        tf.compat.v1.set_random_seed(RANDOM_SEED)
        np.random.seed(seed=RANDOM_SEED)                
        #%% Extract N and M in input
        N=X_np.shape[0]
        M=U_np.shape[0]
        if M<N:
            print("Error: embedding dimension M cannot be smaller then N")
            return
        #%% unitary rigging of X
        RXT_np=quantumgates.riggunitary(X_np,M)
        RXT=tf.constant(RXT_np,dtype=tfdatatype)
        #%% random unitary matrix
        U=tf.constant(U_np,dtype=tfdatatype)        
        #%% generate the training matrix
        W0=tf.random.uniform([M,M],dtype=tfrealdatatype)
        WC=tf.complex(tf.random.uniform([M,M],dtype=tfrealdatatype),tf.random.uniform([M,M],dtype=tfrealdatatype))
        Wreal=tf.compat.v1.get_variable("Wr",initializer=W0,dtype=tfrealdatatype)
        Wimag=tf.compat.v1.get_variable("Wi",initializer=W0,dtype=tfrealdatatype)
        W=tf.compat.v1.get_variable("W",initializer=WC,dtype=tfdatatype,trainable=False)        
        #%% transfer matrix
        transfer_matrix=tf.compat.v1.get_variable("transfer_matrix",initializer=WC,trainable=False)
        #%% current output
        yC=tf.complex(tf.random.uniform([M,1],dtype=tfrealdatatype),tf.random.uniform([M,1],dtype=tfrealdatatype))
        yout=tf.compat.v1.get_variable("current_y",initializer=yC,trainable=False)
        yt=tf.compat.v1.get_variable("target_y",initializer=yC,trainable=False)
        
        #%% place holder
        x=tf.compat.v1.placeholder(dtype=tfdatatype,shape=(M,1),name="x")     
        #%% generate training set
        # it is a single input with M size, first elements are 1 the other are zeros
        xtrains=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(ntrain):
           for i in range(N):
                xtrains[i,j]=1.0
        #%% normalize training set
        #xtrains=tf.keras.utils.normalize(xtrains,axis=0,order=2)         
        #%% generate validation set
        xvalids=np.zeros((M,ntrain),dtype=npdatatype)
        for j in range(nvalid):
           for i in range(M):
                xvalids[i,j]=xtrains[i,j]
        #%% normalize validation set
        #xvalids=tf.keras.utils.normalize(xvalids,axis=0,order=2)             
        #%% equation
        with tf.name_scope("equation") as scope:
            with tf.name_scope("Wreal") as scope:
                variable_summaries(Wreal)
            with tf.name_scope("Wimag") as scope:
                variable_summaries(Wimag)
            yt=tf.matmul(RXT,x)
            #clipping the weigths
            Wreal=tf.clip_by_value(Wreal,realMIN,realMAX)
            Wimag=tf.clip_by_value(Wimag,imagMIN,imagMAX)                       
            # build the matrices (phase only modulator)
            W=tf.complex(tf.cos(Wreal),tf.sin(Wreal))
            transfer_matrix=tf.matmul(U,W)
            yout=tf.matmul(transfer_matrix,x)
            equation=yout-yt
            eqreal=tf.math.real(equation)
            eqimag=tf.math.imag(equation)
            cost_function=tf.reduce_mean(tf.square(eqreal)+
                                         tf.square(eqimag))
            tf.compat.v1.summary.scalar('cost_function',cost_function)
        #%%TO DO : TRY OTHER MINIMIZER
        with tf.name_scope("train") as scope:
           optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
                   cost_function)
        #%% message
        if verbose>0:
            print('Running with M ' + repr(M) +
            ' N ' + repr(N) +
            ' ntrain ' + repr(ntrain) +
            ' nvalid ' + repr(nvalid))                
        #%% writer
        train_writer=tf.compat.v1.summary.FileWriter(tensorboarddir)
        merged=tf.compat.v1.summary.merge_all()
           
        #%%
        xtmp=np.zeros((M,1),dtype=npdatatype)
        with tf.compat.v1.Session()  as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            Tinitial=transfer_matrix.eval()
            for epoch in range(epochs):
                avg_cost=0.
                for i in range(ntrain):
                   xtmp=np.reshape(xtrains[0:M,i],(M,1))
                   sess.run(optimizer,feed_dict={x: xtmp})
                   avg_cost+=sess.run(cost_function, feed_dict={x: xtmp})
                   summary=sess.run(merged, feed_dict={x: xtmp}) 
                   train_writer.add_summary(summary,i+epoch*epochs)
                avg_cost=avg_cost/ntrain
                # messagers
                if epoch % display_steps == 0:
                   # evaluate the validation error
                   avg_cost_valid=0.
                   for i in range(nvalid):
                       xtmp_valid=np.reshape(xvalids[0:M,i],(M,1))
                       avg_cost_valid+=sess.run(cost_function, feed_dict=
                                                {x: xtmp_valid})
                   avg_cost_valid=avg_cost_valid/nvalid
                   if verbose>1:
                       print('epoch '+repr(epoch))
                       print('cost '+repr(avg_cost))
                       print('valid cost '+repr(avg_cost_valid))
                 # check the validation cost and if needed exit the iteration
                if avg_cost_valid < threshold_valid:         
                    if verbose:
                        print('Convergence in validation reached at epoch ' 
                              + repr(epoch))
                    break
                if epoch>=epochs-1:
                    if verbose>0:
                        print('No convergence, maximal epochs reached '
                              +repr(epochs))            
            Tfinal=transfer_matrix.eval()
            ytargetf=sess.run(yt, feed_dict={x: xtmp}) 
            youtf=sess.run(yout, feed_dict={x: xtmp}) 
            rWfinal=Wreal.eval()
            iWfinal=Wimag.eval()
            Wfinal=W.eval()
            TVV=tf.matmul(W,W,adjoint_a=True).eval()
        #    print('Determinant Structure matrix ' + repr(np.linalg.det(dataU_np)))
        #%%
        if verbose>2:
            print("Final Wreal")
            utilities.printonscreennp(rWfinal)    
            print("Final Wimag")
            utilities.printonscreennp(iWfinal)    
            print("Final Sinput=W")
            utilities.printonscreennp(Wfinal)    
            print("Final TV V for unitarity ")
            utilities.printonscreennp(TVV)    
            print("Initial T")
            utilities.printonscreennp(Tinitial)    
            print("Final T")
            utilities.printonscreennp(Tfinal)    
                
        #%%
        sess.close()

        
        #%% set the output dictionary of parameters
        out=dict();
        out['accuracy']=threshold_valid
        out['epoch']=epoch
        out['ntrain']=ntrain
        out['nvalid']=nvalid
        out['N']=X_np.shape[0]
        out['M']=M
        out['X']=X_np
        out['xtrain']=xtrains
        out['Wfinal']=Wfinal
        out['Tfinal']=Tfinal
        out['yt']=ytargetf
        out['y']=youtf
        out['cost']=avg_cost_valid
		
        
        
        return out
                