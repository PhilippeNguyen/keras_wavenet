import keras
import keras.backend as K
from keras.layers import (Conv1D)
import tensorflow as tf
class QueuedConv1D(Conv1D):
    '''For each step in the filter, create a queue of length equal to the dilation rate
        need to add updates somewhere
    '''
    def __init__(self,batch_size=1,**kwargs):
        super(QueuedConv1D,self).__init__(**kwargs)
        self.stateful = True
        self.batch_size = batch_size
    
    def call(self,x):
        #x should have shape num_batch,num_timesteps,num_channels,
        #Specifically, for the queued model, num_timesteps == 1
        x_batch,x_timesteps,n_ch = x._keras_shape
        assert x_timesteps == 1
        assert len(self.dilation_rate) ==1 and len(self.kernel_size) == 1 and len(self.strides) == 1
        assert self.strides[0] == 1
        dilation_rate = self.dilation_rate[0]
        kernel_size = self.kernel_size[0]
        batch_size = self.batch_size
        
        self.queue_list = []
        self.init_list = []  #initialize queues
        self.state_list = [x] #the current states
        self.push_list = [] #queue updates
        
        prev_state = x
        #Set up queue activity
        for filter_step in range(kernel_size - 1):
            q = tf.FIFOQueue(dilation_rate, dtypes=tf.float32, shapes=(batch_size, 1, n_ch))
            init_q = q.enqueue_many(tf.zeros((dilation_rate, batch_size, 1, n_ch)))
            state = q.dequeue()
            push = q.enqueue(prev_state)
            
            self.state_list.append(state)
            prev_state = state
            self.queue_list.append(q)
            self.init_list.append(init_q)
            self.push_list.append(push)
            
        self.add_update(self.push_list)
        self.reset_states()
        
        output = 0
        for filter_step in range(kernel_size):
            kern_slice = self.kernel[filter_step,:,:] #tensor shape (n_in,n_out)
            state = self.state_list[kernel_size-1-filter_step] #states and kernel are in reversed order
            output += K.dot(state,kern_slice) 
        
        if self.use_bias:
            output = K.bias_add(output,self.bias,data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def reset_states(self):
        sess = K.get_session()
        dequeues = []
        for q in self.queue_list:
            q_size = q.size().eval(session=sess)
            deq = q.dequeue_many(q_size)
            dequeues.append(deq)
        sess.run(self.init_list+dequeues)

    def compute_output_shape(self,input_shape):
        x_batch,x_timesteps,n_ch = input_shape
        return (x_batch,x_timesteps,self.filters)

custom_objs = globals()