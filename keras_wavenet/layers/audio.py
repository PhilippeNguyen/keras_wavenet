import numpy as np 
import tensorflow as tf
from keras.layers import Layer
import keras.backend as K

class Spectrogram(Layer):
    
    def __init__(self,S=None, 
             n_fft=2048, hop_length=512,
             power=1.0,freq_format='freq_first',
             abs=True,**kwargs):
        super(Spectrogram, self).__init__(**kwargs)
        self.S = S
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        assert freq_format in ('freq_first','freq_last'), 'freq_format must be either "freq_first" or "freq_last"'
        
        self.freq_format = freq_format
        self.abs = abs
    def call(self,x):
        '''Assumes shape (num_batches,num_time_steps,num_channels)
        '''
        ndim = len(x.get_shape()) 
        if ndim == 3:
            x = tf.transpose(x,perm=[0,2,1])
            if self.freq_format == "freq_first":
                perm = [0,3,2,1]
            else:
                perm = [0,1,2,3]

        elif ndim == 2:
            if self.freq_format == "freq_first":
                perm = [0,2,1]
            else:
                perm = [0,1,2]
        stft = tf.contrib.signal.stft(x,frame_length=self.n_fft,
                          frame_step=self.hop_length,pad_end=True)
        if self.abs:
            stft = tf.abs(stft)
        stft = stft**self.power

        stft = tf.transpose(stft,perm=perm)
        return stft

            
    def compute_output_shape(self,input_shape):
        ndim = len(input_shape)
        n_batches = input_shape[0]
        sig_len = input_shape[1]
        freq_len = self.n_fft//2+1
        time_len = int(np.ceil(sig_len/self.hop_length))
        if ndim ==3:
            if self.freq_format == "freq_first":
                return (n_batches,freq_len,time_len,input_shape[-1])
            else:
                return (n_batches,input_shape[-1],time_len,freq_len)
        elif ndim == 2:
            if self.freq_format == "freq_first":
                return (n_batches,freq_len,time_len)
            else:
                return (n_batches,time_len,freq_len)
        
    def get_config(self):
        config = {'S':self.S,'n_fft':self.n_fft,
                  'hop_length':self.hop_length,'power':self.power,
                  'freq_format':self.freq_format,
                  'abs':self.abs}
        base_config =super(Spectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))