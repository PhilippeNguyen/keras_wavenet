import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from keras_wavenet.models.audio_outputs import OutputProcessor
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,Input,Conv1D,Activation,Lambda
import numpy as np

class DMLL_tfp(OutputProcessor):
    '''discretized mixture of logistics
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/distributions/python/ops/quantized_distribution.py
    '''

    def __init__(self,num_classes,
                 mean_scale=20000,
                 scale_scale=0.1,
                 low=0.,high=2**16 - 1.,
                 log_scale_min=-7.0,log_scale_max=8.0,
                 **kwargs):
        super(DMLL_tfp,self).__init__(**kwargs)
        self.num_classes = num_classes
        self.built=False
        self.low = low
        self.high = high
        self.mean_scale = mean_scale
        self.scale_scale = scale_scale
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.sess = K.get_session()
        
    def loss(self,y_true,y_pred,fix_y_true=True):
        '''Args:
                y_true : (nbatches,num_timesteps,1) with range [0,2**16 - 1]
                y_pred :(nbatches,num_timesteps,3*self.num_classes) with the final
                    dimension being: 0:num_classes --> class coefficient (pi)
                                     num_classes:2*num_classes --> class means
                                     2*num_classes:3*num_classes --> class 
        '''
        if not self.built:
            self._build(y_pred)
        
        y_true = y_true[...,0]
        if fix_y_true:
            y_true = K.minimum(K.maximum(y_true,self.low),self.high)
        
        neg_log_likelihood = -K.sum(self.dist.log_prob(y_true),axis=-1)
        return neg_log_likelihood
        
    def sample(self,probs,model):
        if not self.built:
            self._build(model.output)
        
        return np.expand_dims(
                self.sess.run(
                    fetches=[self.sample_op],feed_dict={model.output:probs})[0],
                axis=-1)
    
    def _build(self,model_output):
        assert model_output.get_shape().as_list()[-1] % self.num_classes == 0
        logit_probs = model_output[...,:self.num_classes]
        means = model_output[...,self.num_classes:2*self.num_classes]
        scale_params = model_output[...,2*self.num_classes:]
        
        means = means*self.mean_scale
        
        log_scales = K.minimum(K.maximum(scale_params* self.scale_scale,
                                         self.log_scale_min),
                            self.log_scale_max)
        scales = K.exp(log_scales) 

        discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=means, scale=scales),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=self.low,
                high=self.high)
        self.dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logit_probs),
                components_distribution=discretized_logistic_dist)
        self.sample_op = self.dist.sample()
        self.built=True
        
        
class SparseCategorical_tfp(OutputProcessor):

    def __init__(self,num_classes,
                     bias,
                     mode='logits',
                 **kwargs):
        super(DMLL_tfp,self).__init__(**kwargs)
        self.num_classes = num_classes
        self.built=False
        self.bias = bias
        assert mode in ('logits','probs')
        self.mode=mode
        self.sess = K.get_session()
        
    def loss(self,y_true,y_pred):
        '''Args:
        '''
        if not self.built:
            self._build(y_pred)
        
        y_true = y_true[:,:,0] + self.bias
        
        neg_log_likelihood = -K.sum(self.dist.log_prob(y_true),axis=-1)
        return neg_log_likelihood
        
    def sample(self,probs,model):
        if not self.built:
            self._build(model.output)
        
        return np.expand_dims(
                self.sess.run(
                    fetches=[self.sample_op],feed_dict={model.output:probs})[0],
                axis=-1)
    
    def _build(self,model_output):
        if self.mode == 'logits':
            self.dist = tfd.Categorical(logits=model_output)
        elif self.mode == 'probs':
            self.dist = tfd.Categorical(probs=model_output)
        self.sample_op = self.dist.sample()
        self.built=True

class Logistic_tfp(OutputProcessor):
    def __init__(self,num_classes=1,
                 **kwargs):
        super(DMLL_tfp,self).__init__(**kwargs)
        self.num_classes = num_classes
        self.built=False

        self.sess = K.get_session()
        
    def loss(self,y_true,y_pred,):
        '''Args:
            '''
        if not self.built:
            self._build(y_pred)
        
        y_true = y_true[...,0]
        
        neg_log_likelihood = -K.sum(self.dist.log_prob(y_true),axis=-1)
        return neg_log_likelihood
        
    def sample(self,probs,model):
        if not self.built:
            self._build(model.output)
        
        return np.expand_dims(
                self.sess.run(
                    fetches=[self.sample_op],feed_dict={model.output:probs})[0],
                axis=-1)
    
    def _build(self,model_output):
        assert model_output.get_shape().as_list()[-1] == 2
        scale = model_output[...,0]
        shift = model_output[...,1]
        self.dist = tfd.Logistic(loc=shift,scale=scale)
        
        self.sample_op = self.dist.sample()
        self.built=True 
    
if __name__ == '__main__':
    num_classes = 10
    data = np.floor(np.random.uniform(0,2**16-1,size=(10000,10,1)))
    

    
    input_layer = Input(shape=data.shape[1:])
    x = Lambda(lambda x : x/(2**16-1))(input_layer)
    x = Conv1D(6*num_classes,kernel_size=1)(x)
    x = Activation('relu')(x)
    output = Conv1D(3*num_classes,kernel_size=1)(x)
    model = tf.keras.Model(input_layer,output)
    
    output_processor = DMLL_tfp(num_classes)
    loss = output_processor.loss
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=loss)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=10)
    model.fit(data,data,epochs=1000,validation_split=0.8,callbacks=[early_stop])
    sample = output_processor.sample(model.predict(data),model,)
