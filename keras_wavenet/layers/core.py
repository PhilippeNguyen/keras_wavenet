import keras
import keras.backend as K
from keras.layers import (Layer,Reshape,Add,Conv1D,SeparableConv1D,
                          Activation,GlobalMaxPool1D,MaxPool1D,Flatten,Lambda)

import numpy as np


##############
### Layers ###
##############

class TemporalShift(Layer):
    """ shift a temporal tensor (n_batch,time,channels)
        
        Args:
            shift: int, number of timesteps to shift, adds zeros to the front if postivem
                removes indices from the front if negative.
    """
    
    def __init__(self,shift,**kwargs):
        super(TemporalShift,self).__init__(**kwargs)
        self.shift = shift
        self.padding = (np.maximum(shift,0),-np.minimum(shift,0))
        
    def call(self,x):
        x_shape = x._keras_shape
        assert len(x_shape) == 3, "TemporalShift input must be dim 3"
        x = K.temporal_padding(x, padding=self.padding)
        return x[:,:x_shape[1]]
        
    def get_config(self):
        config = {'shift':self.shift}
        base_config = super(TemporalShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class WavenetActivation(Layer):
    '''Takes the first half channels and applies sigmoid,
        takes the last half channels and applies tanh
        then elementwise multiply the two.
        
        This lets you obtain the sigmoid and tanh tensors without doing the convolutions
        separately
    '''
    def call(self,x):
        _,_,n_ch = x._keras_shape
        half_ch = n_ch//2
        x_sigmoid = K.sigmoid(x[:, :, :half_ch])
        x_tanh = K.tanh(x[:, :, half_ch:])
        return x_sigmoid * x_tanh
    def compute_output_shape(self,input_shape):
        assert len(input_shape) == 3
        n_b,n_time,n_ch = input_shape
        assert n_ch %2 == 0
        half_ch = n_ch //2
        return (n_b,n_time,half_ch)
    
class RepeatElements(Layer):
    def __init__(self,num_repeats,axis,**kwargs):
        super(TemporalShift,self).__init__()
        self.num_repeats = num_repeats
        self.axis = axis
        
    def call(self,x):
        return K.repeat_elements(x,rep=self.num_repeats,axis=self.axis)
    
    def compute_output_shape(self,input_shape):
        input_shape = list(input_shape)
        input_shape[self.axis-1] = input_shape[self.axis-1]*self.num_repeats
        return tuple(input_shape)
    
    def get_config(self):
        config = {'num_repeats':self.num_repeats,
                  'axis':self.axis}
        base_config = super(TemporalShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AddEncoder(Layer):
    '''Adds the encoder to the signal,
        function call AddEncoder()([signal,encoding])
    '''
    def __init__(self,**kwargs):
        super(AddEncoder,self).__init__(**kwargs)
    
    def call(self,inputs):
        assert isinstance(inputs,(list,tuple))
        assert len(inputs) == 2 
        
        sig,enc = inputs
        sig_nb, sig_len, sig_ch = sig._keras_shape
        enc_nb, enc_len, enc_ch = enc._keras_shape
        
        assert sig_nb == enc_nb
        assert sig_ch == enc_ch
        
        sig_re = K.reshape(sig, (K.shape(sig)[0],)+(enc_len,-1,enc_ch))
        intermediate_shape = self._fix_unknown_dimension(sig._keras_shape[1:],
                                                         (enc_len,-1,enc_ch))
        intermedia_len = intermediate_shape[1]
        
        enc_re = K.reshape(enc, (K.shape(enc)[0],)+(enc_len,1,enc_ch))
        enc_re=K.repeat_elements(enc_re,rep=intermedia_len,axis=2)
        
        added_re = sig_re + enc_re
        
        added = K.reshape(added_re,(K.shape(sig)[0],)+(sig_len,enc_ch))
        return added
    
    def compute_output_shape(self,input_shape):
        return input_shape[0]
    
    def _fix_unknown_dimension(self, input_shape, output_shape):
        """See keras.layers.Reshape
        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)



custom_objs = globals()