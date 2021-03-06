from keras_wavenet.models.wavenet import build_wavenet_decoder,build_noskip_wavenet_decoder
from keras.layers import Conv1D,Add,Multiply,Lambda,Activation
import keras.backend as K
import numpy as np
import keras


def wavenet_iaf_step(signal,encoding,
                     width,skip_width,out_width,
                     num_layers,num_stages,
                     filt_len=3,log_scale_min=-5.0,log_scale_max=5.0,
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
    log_scale_pre_sig = Conv1D(filters=1,
            kernel_size=1,
            name=base_name+'_iaf_logscale_pre_sig')(dec_out)
    log_scale_sig = Activation('sigmoid',name=base_name+'_iaf_logscale_sig')(log_scale_pre_sig)
    log_range = log_scale_max-log_scale_min
    log_scale = Lambda(lambda x : (log_range)*x + log_scale_min,
                       name=base_name+'_iaf_logscale')(log_scale_sig)
    scale = Activation('exponential',name=base_name+'_iaf_scale')(log_scale)
    new_x = keras.layers.Multiply()([signal,scale])
    new_x = keras.layers.Add()([new_x,shift])
    return new_x,shift,scale,log_scale

def wavenet_noskip_iaf_step(signal,encoding,
                     width,out_width,
                     num_layers,num_stages,
                     filt_len=3,log_scale_min=-5.0,log_scale_max=5.0,
                     base_name=""):
    #NOTE: the parallel wavenet says to remove skip connections...
    dec_out = build_noskip_wavenet_decoder(signal,encoding,
                                    width,out_width,
                                    num_layers,num_stages,
                                    filt_len=filt_len,
                                    final_conditioning=True,
                                    final_activation='relu',
                                    base_name=base_name)
    
    shift = Conv1D(filters=1,
       kernel_size=1,
       name=base_name+'_iaf_shift')(dec_out)
    log_scale_pre_sig = Conv1D(filters=1,
            kernel_size=1,
            name=base_name+'_iaf_logscale_pre_sig')(dec_out)
    log_scale_sig = Activation('sigmoid',name=base_name+'_iaf_logscale_sig')(log_scale_pre_sig)
    log_range = log_scale_max-log_scale_min
    log_scale = Lambda(lambda x : (log_range)*x + log_scale_min,
                       name=base_name+'_iaf_logscale')(log_scale_sig)
    scale = Activation('exponential',name=base_name+'_iaf_scale')(log_scale)
    new_x = keras.layers.Multiply()([signal,scale])
    new_x = keras.layers.Add()([new_x,shift])
    return new_x,shift,scale,log_scale

