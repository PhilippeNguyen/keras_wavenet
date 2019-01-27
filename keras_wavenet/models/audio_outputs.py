import keras
import keras.backend as K
from keras.losses import sparse_categorical_crossentropy
import numpy as np
from keras.activations import sigmoid,softplus
class OutputProcessor(object):
    def __init__(self):
        pass
    def loss():
        pass
    def sample():
        pass
    
class SparseCategorical(OutputProcessor):
    '''For wavenet num_classes = 257,
        biase  =128
    '''
    def __init__(self,num_classes,bias=128,**kwargs):
        super(SparseCategorical,self).__init__(**kwargs)
        self.num_classes = num_classes
        self.bias = bias
    def loss(self,y_true,y_pred):
        y_true = K.cast(K.batch_flatten(y_true), 'int32') + self.bias
        return K.sum(sparse_categorical_crossentropy(y_true,y_pred),axis=-1)
    def sample(self,probs,axis=-1):
        batch_size,n_timesteps,_ = probs.shape
        assert n_timesteps == 1
       
        probs = probs[:,0,:]
        cdf = np.cumsum(probs, axis=1)
        rand_vals = np.random.rand(batch_size)
        idxs = np.zeros([batch_size, 1],dtype='int64')
        for i in range(batch_size):
            idxs[i] = cdf[i].searchsorted(rand_vals[i])
        return np.expand_dims(idxs,-1) - self.bias
    
def get_output_processor(processor_name,num_output_channels,processor_kwargs=None):
    if processor_kwargs is None:
        processor_kwargs = {}
    if processor_name.lower() in ('sparse_categorical','sparsecategorical'):
#        from keras_wavenet.models.audio_outputs import SparseCategorical
        output_processor = SparseCategorical(num_classes=num_output_channels,
                                         **processor_kwargs)
    elif processor_name.lower() == 'dmll_tfp':
        from keras_wavenet.models.audio_outputs_tf import DMLL_tfp
        assert num_output_channels % 3 == 0, "output_channels must be divisible by 3"
        output_processor = DMLL_tfp(num_classes=num_output_channels//3,
                                **processor_kwargs)
    elif processor_name.lower() in ('sparse_categorical_tfp','sparsecategorical_tfp'):
        from keras_wavenet.models.audio_outputs_tf import SparseCategorical_tfp
        output_processor = SparseCategorical_tfp(num_classes=num_output_channels,
                        **processor_kwargs)
    elif processor_name.lower() in ('logistic_tfp'):
        from keras_wavenet.models.audio_outputs_tf import Logistic_tfp
        output_processor = Logistic_tfp(num_classes=num_output_channels//2,
                        **processor_kwargs)
    return output_processor
