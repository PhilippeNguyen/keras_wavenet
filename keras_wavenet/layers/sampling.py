import keras
from keras.layers import Layer
import keras.backend as K
import numpy as np
   
    
class GaussianSampling(Layer):

    def __init__(self,mean=0.,stddev=1.,**kwargs):
        super(GaussianSampling,self).__init__(**kwargs)
        self.mean = mean
        self.stddev = stddev

            
    def call(self,inputs):
        return K.random_normal(shape=K.shape(inputs),
                                    mean=self.mean, stddev=self.stddev)
    
    def get_config(self):
        config = {'mean':self.mean,'stddev':self.stddev}
        base_config = super(GaussianSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self,input_shape):
        return input_shape
    
if __name__ == '__main__':
    data = np.zeros((10,100,1))
    
#    input_layer = keras.layers.Input(shape=data.shape[1:])
#    output_layer = GaussianSampling()(input_layer)
#    model = keras.models.Model(input_layer,output_layer)
#    bb = model.predict(data)
    
    rand = K.random_normal(data.shape)
    data_1 = np.ones_like(data)
    input_layer_rand = keras.layers.Input(tensor=rand)
    input_layer_1 = keras.layers.Input(shape=data_1.shape[1:])
    x = keras.layers.Add()([input_layer_rand,input_layer_1])
    model = keras.models.Model([input_layer_rand,input_layer_1],x)
    bb = model.predict(data_1)