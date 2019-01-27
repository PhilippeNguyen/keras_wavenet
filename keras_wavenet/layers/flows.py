from keras.layers import Layer,InputSpec
import keras.backend as K
from keras import activations


class Distribution(Layer):
    '''Keras layer to help with sampling
    '''
    def __init__(self,**kwargs):
        super(Distribution,self).__init__(**kwargs)
    def call(self,inputs):
        '''Calling a Distribution should return a tensor of samples from the 
            distribution
        '''
        pass
class Gaussian(Distribution):
    '''Represents an isotropic gaussian distribution.
        For numerical stability reasons, we parameterize z with the log variance.
            Though we use standard deviation for the prior
        Also lets you define the parameters for a gaussian prior so that you
        can compute the kl divergence
    '''
    def __init__(self,prior_mean=0.,prior_stddev=1.,**kwargs):
        super(Gaussian,self).__init__(**kwargs)
        self.prior_mean = prior_mean
        if prior_stddev <= 0:
            raise ValueError("prior_stddev must be > 0")
        self.prior_stddev = prior_stddev

            
    def call(self,inputs):
        '''
            Args: inputs, should be a list of tensors representing the [mean,log_variance] 
                    of your input distribution
        '''
        z_mean, z_log_var = inputs
        
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]),
                                    mean=0, stddev=1)
        return z_mean + K.exp(z_log_var/2) * epsilon
    
    def get_config(self):
        config = {'prior_mean':self.prior_mean,'prior_stddev':self.prior_stddev}
        base_config = super(Gaussian, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def kl_divergence(self,inputs):
        z_mean, z_log_var = inputs
        kl_loss = K.sum((   
                      K.log(self.prior_stddev) 
                    - 0.5*z_log_var 
                    + (K.exp(z_log_var) + K.square(z_mean-self.prior_mean))/(2*K.square(self.prior_stddev)) 
                    - 0.5
                   ),axis=-1)
        return kl_loss
    
    def compute_output_shape(self,input_shape):
        assert len(input_shape) == 2
        return input_shape
    
class Flow(Layer):
    '''Represents a flow
    ''' 
    def __init__(self,**kwargs):
        super(Flow,self).__init__(**kwargs)
    def build(self):
        pass
    def call(self,inputs):
        pass
    def compute_output_shape(self,input_shape):
        pass
    def logdetjac(self,inputs):
        '''Note that the layer must be "built" before the logdetjac can be calculated.
            this means it must have been called on an input first.
        '''
        pass
    def apply(self,inputs):
        '''Given an appropriate set of inputs, applies both the flow call
            and returns the logdetjac
        '''
        out_tensor = self.__call__(inputs)
        logdetjac = self.logdetjac(inputs)
        return out_tensor,logdetjac


'''The two flows as described in (Rezende et al 2016) https://arxiv.org/pdf/1505.05770.pdf
    The Det-jacs for these two flows are computable in O(d), where d is the latent dimensionality.
'''
class PlanarFlow(Flow):
    '''Transforms a distribution by making circles/holes
    '''
    def __init__(self,activation='tanh',
                 u_initializer='glorot_uniform',
                 w_initializer='glorot_uniform',
                 b_initializer='glorot_uniform',
                 **kwargs):
        super(PlanarFlow,self).__init__(**kwargs)
        self.u_initializer = u_initializer 
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer
        self.h = activations.get(activation)
        self.activation = activation
        
    def build(self,input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]
        
        self.u = self.add_weight(shape=(input_dim,1),
                              initializer=self.u_initializer,
                              name='u',)
        self.w = self.add_weight(shape=(input_dim,1),
                              initializer=self.w_initializer,
                              name='w',)
        self.b = self.add_weight(shape=(1,1),
                          initializer=self.b_initializer,
                          name='b',)
        self.input_spec = InputSpec(ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self,inputs):
        wzb = K.batch_dot(inputs,self.w) + self.b
        return inputs+ K.transpose(self.u)*self.h(wzb)
    
    def logdetjac(self,inputs):
        wzb = K.stop_gradient(K.batch_dot(inputs,self.w) + self.b)
        psi = K.batch_dot(K.gradients(self.h(wzb),wzb)[0],self.w)
        return K.log(K.abs(1+K.batch_dot(psi,self.u)))
    

    
    def get_config(self):
        config = {'u_initializer': self.u_initializer,
                  'w_initializer': self.w_initializer,
                  'b_initializer': self.b_initializer,
                  'activation': self.activation}
        base_config = super(PlanarFlow, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,input_shape):
        assert len(input_shape) == 2
        return input_shape

class RadialFlow(Flow):
    '''Transforms a distribution by making circles/holes
    '''
    def __init__(self,
                 z_mean_initializer='glorot_uniform',
                 alpha_initializer='glorot_uniform',
                 beta_initializer='zeros',
                 **kwargs):
        '''Note we apply softplus to alphmapsa since it should be non-negative
        '''
        super(RadialFlow,self).__init__(**kwargs)
        self.z_mean_initializer = z_mean_initializer 
        self.alpha_initializer = alpha_initializer
        self.beta_initializer = beta_initializer
        
    def build(self,input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.z_mean = self.add_weight(shape=(input_dim,),
                              initializer=self.z_mean_initializer,
                              name='z_mean',)
        self.alpha = self.add_weight(shape=(1,),
                              initializer=self.alpha_initializer,
                              name='alpha',)
        self.beta = self.add_weight(shape=(1,),
                          initializer=self.beta_initializer,
                          name='beta',)
        self.input_spec = InputSpec(ndim=2, axes={-1: input_dim})
        self.built = True
    
    def _r(self,z):
        r = K.sqrt(K.sum(K.square(z-self.z_mean)))
        return r
    
    def _h(self,z):
        alpha = activations.softplus(self.alpha)
        return 1./(alpha+self._r(z))
    
    #no need to use autograd for h'
    def _hn(self,z):
        alpha = activations.softplus(self.alpha)
        return -1./K.square((alpha+self._r(z)))
    
    def call(self,inputs):
        z_diff = inputs - self.z_mean
        return inputs + self.beta*self._h(inputs)*z_diff
    
    def logdetjac(self,inputs):
        expd = 1 + self.beta*self._h(inputs)
        return K.log(K.pow(expd,self.input_dim-1)*(expd+self.beta*self._hn(inputs)*self._r(inputs)))
    
    def get_config(self):
        config = {'z_mean_initializer': self.z_mean_initializer,
                  'alpha_initializer': self.alpha_initializer,
                  'beta_initializer': self.beta_initializer,
                  }
        base_config = super(RadialFlow, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,input_shape):
        assert len(input_shape) == 2
        return input_shape
    

    