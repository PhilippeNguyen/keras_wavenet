from keras.layers import (Reshape,Add,Conv1D,SeparableConv1D,
                          Activation,GlobalMaxPool1D,MaxPool1D,Flatten,Lambda)
from ..layers.core import WavenetActivation,AddEncoder,TemporalShift

##############
### Blocks ###
##############

def res_block(sig,width,filt_len,dilation,base_name,
              encoding=None,conv_type='conv1d',separate_skip=True,
              skip_width=None):
    
    if skip_width == None:
        skip_width = width
    
    filt_sig = Conv1D(filters=2*width,
               kernel_size=filt_len,
               dilation_rate=dilation,
               padding='causal',
               name=base_name+"_sigconv_1")(sig)
    
    if encoding is not None:
        enc_wide = Conv1D(filters=2*width,
                   kernel_size=filt_len,
                   dilation_rate=dilation,
                   padding='causal',
                   name=base_name+"_encconv_1")(encoding)
        filt_sig = AddEncoder()([filt_sig,enc_wide])
        
    filt_sig = WavenetActivation()(filt_sig)
    
    last_conv = Conv1D(filters=width,kernel_size=1,
                      name=base_name+"_lastconv",
                      padding='causal')(filt_sig) #technically causal padding doesnt matter here
    if separate_skip:
            skip = Conv1D(filters=skip_width,kernel_size=1,
                      name=base_name+"_sepskipconv",
                      padding='causal')(filt_sig)
    else:
        skip = last_conv
            
    out_res = Add()([sig,last_conv])
    return out_res,skip

##############
### Models ###
##############
    
def build_wavenet_encoder(stft_input,width,filter_len,num_layers,num_stages,
                          pool_size=None
                          ):
    
    en = Conv1D(filters=width,kernel_size=filter_len,padding='same')(stft_input)
    
    for num_layer in range(num_layers):
        dilation = 2**(num_layer % num_stages)
        d = Activation('relu')(en)
        d = Conv1D(filters=width,kernel_size=filter_len,padding='same',
                   dilation_rate=dilation)(en)
        d = Activation('relu')(d)
        
        en_out = Conv1D(filters=width,kernel_size=1)(d)
        en = Add()([en,en_out])
    en = Conv1D(filters=width,kernel_size=1)(en)
    

    return en

def build_wavenet_decoder(encoding,signal,
                          width,skip_width,out_filters,
                          num_layers,num_stages,
                          filt_len=3,final_conditioning=False):
    sig_shift = TemporalShift(shift=1)(signal)
    _,sig_len,_ = signal._keras_shape
    
    sig = Conv1D(filters=width,
                   kernel_size=filt_len,
                   padding='causal',
                   name="convstart")(sig_shift)
    skip = Conv1D(filters=skip_width,
                   kernel_size=filt_len,
                   padding='causal',
                   name="skipstart_conv")(sig_shift)
    skip_list = [skip]
    for idx in range(num_layers):
        dilation = 2**(idx % num_stages)
        sig,new_skip = res_block(sig,width,filt_len,dilation=dilation,
                                 skip_width=skip_width,
                                 base_name='res_block'+str(idx)+'_dil'+str(dilation),
                                 encoding=encoding,
                                 conv_type='conv1d')
        skip_list.append(new_skip)
    skip_add = Add()(skip_list)
    skip_add = Activation('relu')(skip_add)

    if final_conditioning:
        skip_conv = Conv1D(filters=width,
               kernel_size=1,
               padding='causal',
               name="skip_conv")(skip_add)
        conv_encoding = Conv1D(filters=width,
                       kernel_size=1,
                       padding='causal',
                       name="conv_encoding")(encoding)
        skip_conditioned = AddEncoder()([skip_conv,conv_encoding])
        skip_cond_conv = Conv1D(filters=out_filters,
                       kernel_size=1,
                       padding='causal',
                       name="skip_cond_conv")(skip_conditioned)
        return skip_cond_conv
    else:
        skip_conv = Conv1D(filters=out_filters,
               kernel_size=1,
               padding='causal',
               name="skip_conv")(skip_add)
        return skip_conv
    