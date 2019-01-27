from keras_wavenet.models.wavenet import build_wavenet_decoder
from keras.layers import Conv1D,Add,Multiply,Lambda,Activation
import keras.backend as K
import numpy as np
import keras


def wavenet_iaf_step(signal,encoding,
                     width,skip_width,out_width,
                     num_layers,num_stages,
                     filt_len=3,
                     base_name=""):
    #NOTE: the parallel wavenet says to remove skip connections...
    dec_out = build_wavenet_decoder(signal,encoding,
                                    width,skip_width,out_width,
                                    num_layers,num_stages,
                                    filt_len=filt_len,
                                    final_conditioning=True,
                                    final_activation='relu',
                                    base_name=base_name)
    
    shift = Conv1D(filters=1,
       kernel_size=1,
       name=base_name+'_iaf_shift')(dec_out)
    log_scale_unclipped = Conv1D(filters=1,
       kernel_size=1,
       name=base_name+'_iaf_logscale_unclipped')(dec_out)
    log_scale = Lambda(lambda x : K.minimum(K.maximum(x,-7.0),7.0),
                       name=base_name+'_iaf_logscale')(log_scale_unclipped)
    scale = Activation('exponential',name=base_name+'_iaf_scale')(log_scale)
    new_x = keras.layers.Multiply()([signal,scale])
    new_x = keras.layers.Add()([new_x,shift])
    return new_x,shift,scale,log_scale

