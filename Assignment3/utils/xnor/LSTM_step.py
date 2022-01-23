import tensorflow as tf
import numpy as np

def LSTM_step(cell_inputs, cell_states, kernel, recurrent_kernel, bias):
    """
    Run one time step of the cell. That is, given the current inputs and the cell states from the last time step, calculate the current state and cell output.
    You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the very basic LSTM functionality.
    Hint: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.
        
        
    :param cell_inputs: The input at the current time step. The last dimension of it should be 1.
    :param cell_states:  The state value of the cell from the last time step, containing previous hidden state h_tml and cell state c_tml.
    :param kernel: The kernel matrix for the multiplication with cell_inputs
    :param recurrent_kernel: The kernel matrix for the multiplication with hidden state h_tml
    :param bias: Common bias value
    
    
    :return: current hidden state, and a list of hidden state and cell state. For details check TensorFlow LSTMCell class.
    """
    
    
    ###################################################
    # TODO:      INSERT YOUR CODE BELOW               #
    # params                                          #
    ###################################################
    # Hint: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.
    # cell_inputs = np.ones((1,1))
    # cell_states = [0.2*np.ones((1,64)), np.zeros((1,64))]
    # kernel = 0.1*np.ones((1,256))
    # recurrent_kernel = 0.1*np.ones((64,256))
    # bias = np.zeros(256) Common bias value
    
    # Decompose the kernel matrix for the multiplication with cell_inputs
    k=int(kernel.shape[1]/4)
    K1=kernel[:,:k]
    K2=kernel[:,k:2*k]
    K3=kernel[:,2*k:3*k]
    K4=kernel[:,3*k:4*k]
    
    # Decompose the kernel matrix for the multiplication with hidden state h_tml
    rk=int(recurrent_kernel.shape[1]/4)
    RK1=recurrent_kernel[:,:rk]
    RK2=recurrent_kernel[:,rk:2*rk]
    RK3=recurrent_kernel[:,2*rk:3*rk]
    RK4=recurrent_kernel[:,3*rk:4*rk]
    
    # Decompose common bias value
    b=int(bias.shape[0]/4)
    Bf=bias[:b]
    Bi=bias[b:2*b]
    Bc=bias[2*b:3*b]
    Bo=bias[3*b:4*b]
    
    # Concatenate h and input x, h has shape [1,64], x has shape [1,1], concatenated matrix has shape [1,65]
    prev_ht=cell_states[0]
    xt=cell_inputs
    prev_ht_xt=np.concatenate((prev_ht, xt),axis=1)
    
    # Concatenate recurrent_kernel and kernel to build W, W has shape [65,64]
    Wf=np.concatenate((RK1,K1))
    Wi=np.concatenate((RK2,K2))
    Wc=np.concatenate((RK2,K2))
    Wo=np.concatenate((RK2,K2))
    
    # Forget gate
    ForgetG=tf.math.sigmoid(tf.constant(prev_ht_xt.dot(Wf)),float)+Bf
    
    
    # Input gate
    InputG=tf.math.sigmoid(tf.constant(prev_ht_xt.dot(Wi)),float)+Bi
    
    # Candidate cell state
    CandidateC=tf.math.tanh(tf.constant(prev_ht_xt.dot(Wc)),float)+Bc
    
    # Cell state
    prev_Ct=cell_states[1]
    Ct=tf.math.multiply(ForgetG, prev_Ct)+tf.math.multiply(InputG,CandidateC)
    
    # Output Gate
    OutputG=tf.math.sigmoid(tf.constant(prev_ht_xt.dot(Wo)),float)+Bo
    
    # Output
    ht=tf.math.multiply(OutputG,tf.math.tanh(Ct))
    
    cur_cell_states=[ht,Ct]
    
#     print('kernel shape:',kernel.shape)
#     print('decomposed kernel shape:',K1.shape,K2.shape,K3.shape,K4.shape)
#     print('recurrent_kernel.shape:',recurrent_kernel.shape)
#     print('decomposed recurrent_kernel shape:',RK1.shape,RK2.shape,RK3.shape,RK4.shape)
#     print('bias shape:',bias.shape)
#     print('decomposed bias shape:',Bf.shape,Bi.shape,Bc.shape,Bo.shape)
#     print('prev_ht_xt.shape:',prev_ht_xt.shape)
#     print('Wi shape:',Wi.shape,'Wf shape:',Wf.shape,'Wc shape:',Wc.shape,'Wo shape:',Wo.shape)
#     print('ForgetG shape:',ForgetG.shape)
#     print('InputG shape:',InputG.shape)
#     print('CandidateC shape:',CandidateC.shape)
#     print('Ct.shape:',Ct.shape)
#     print('OutputG shape:',OutputG.shape)
#     print('ht.shape:',ht.shape)
    return cur_cell_states[0], cur_cell_states
    ###################################################
    # END TODO                                        #
    ###################################################
