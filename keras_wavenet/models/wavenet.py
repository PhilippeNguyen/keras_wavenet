
from keras.layers import (Reshape,Add,Conv1D,SeparableConv1D,
                          Activation,GlobalMaxPool1D,MaxPool1D,Flatten,Lambda)
from keras_wavenet.layers.wavenet import WavenetActivation,AddEncoder,TemporalShift

##############
### Blocks ###
##############

def res_block(sig,width,filt_len,dilation,base_name,
              encoding=None,separate_skip=True,
              skip_width=None):
    '''Encompasses the core building block of the wavenet model, see figure 4 
    of the wavenet paper (https://arxiv.org/pdf/1609.03499.pdf)
    
        Args:
            sig: tensor, signal tensor,
            width: int, number of filters/units used in conv operations
            filter_len: int, length of the conv operatations
            dilation: int, dilation rate used in conv
            base_name: name for the keras layers/tensors
            encoding: tensor, encoding tensor,
            separate_skip: bool, whether to have the skip output be separate 
                from the main channel output
            skip_width: int, number of filters/units used in the skip output 
                (output of each residual block)
    '''
    
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

def noskip_block(sig,width,filt_len,dilation,base_name,
              encoding=None,):
    '''Encompasses the core building block of the wavenet model, see figure 4 
    of the wavenet paper (https://arxiv.org/pdf/1609.03499.pdf)
    
        Args:
            sig: tensor, signal tensor,
            width: int, number of filters/units used in conv operations
            filter_len: int, length of the conv operatations
            dilation: int, dilation rate used in conv
            base_name: name for the keras layers/tensors
            encoding: tensor, encoding tensor,
    '''
    
    
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
            
    out_res = Add()([sig,last_conv])
    return out_res

##############
### Models ###
##############
    
def build_wavenet_encoder(input_tensor,width,filter_len,
                          num_layers,num_stages
                          ):
    '''
        Args:
            input_tensor: tensor input, shape (num_batches,num_timesteps,num_channels)
            width: int, number of filters/units used in conv operations
            filter_len: int, length of the conv operatations
            num_layers: int, number of conv stages, each one doubles the dilation rate
            num_stages: int, after every num_stages, reset the dilation rate,
        
    '''
    en = Conv1D(filters=width,kernel_size=filter_len,padding='same')(input_tensor)
    
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

def build_wavenet_decoder(signal,encoding,
                          width,skip_width,out_width,
                          num_layers,num_stages,
                          filt_len=3,final_conditioning=False,
                          final_activation='softmax',
                          base_name=""):
    '''
        Args:
            encoding: tensor input, shape (num_batches,num_encoding_timesteps,num_channels)
            signal: tensor, should be the same as the encoding model input, shape (num_batches,num_timesteps,num_channels)
            width: int, number of filters/units used in the main channel
            skip_width: int, number of filters/units used in the skip output 
            out_width: int, number of filters/units for the final output of the decoder
            filter_len: int, length of the conv operatations
            num_layers: int, number of conv stages, each one doubles the dilation rate
            num_stages: int, after every num_stages, reset the dilation rate,
            final_conditioning: bool, if true conditions the output of the res blocks
            final_activation : str, name of the final activation function
    '''
    
    sig_shift = TemporalShift(shift=1,name=base_name+'temporal_shift')(signal)
    _,sig_len,_ = signal._keras_shape
    
    sig = Conv1D(filters=width,
                   kernel_size=filt_len,
                   padding='causal',
                   name=base_name+"convstart")(sig_shift)
    skip = Conv1D(filters=skip_width,
                   kernel_size=filt_len,
                   padding='causal',
                   name=base_name+"skipstart_conv")(sig_shift)
    skip_list = [skip]
    for idx in range(num_layers):
        dilation = 2**(idx % num_stages)
        sig,new_skip = res_block(sig,width,filt_len,dilation=dilation,
                                 skip_width=skip_width,
                                 base_name=base_name+'res_block'+str(idx)+'_dil'+str(dilation),
                                 encoding=encoding,
                                 )
        skip_list.append(new_skip)
    skip_add = Add()(skip_list)
    skip_add = Activation('relu')(skip_add)

    if final_conditioning:
        skip_conv = Conv1D(filters=width,
               kernel_size=1,
               padding='causal',
               name=base_name+"skip_conv")(skip_add)
        conv_encoding = Conv1D(filters=width,
                       kernel_size=1,
                       padding='causal',
                       name=base_name+"conv_encoding")(encoding)
        skip_conditioned = AddEncoder()([skip_conv,conv_encoding])
        skip_conditioned = Activation('relu')(skip_conditioned)
        skip_out = Conv1D(filters=out_width,
                       kernel_size=1,
                       padding='causal',
                       name=base_name+"decoder_out_pre_act")(skip_conditioned)
    else:
        skip_out = Conv1D(filters=out_width,
               kernel_size=1,
               padding='causal',
               name=base_name+"decoder_out_pre_act")(skip_add)
    decoder_out = Activation(final_activation,name=
                         base_name+'decoder_out')(skip_out)
    return decoder_out


def build_noskip_wavenet_decoder(signal,encoding,
                          width,out_width,
                          num_layers,num_stages,
                          filt_len=3,final_conditioning=False,
                          final_activation='softmax',
                          base_name=""):
    '''
        Args:
            encoding: tensor input, shape (num_batches,num_encoding_timesteps,num_channels)
            signal: tensor, should be the same as the encoding model input, shape (num_batches,num_timesteps,num_channels)
            width: int, number of filters/units used in the main channel
            skip_width: int, number of filters/units used in the skip output 
            out_width: int, number of filters/units for the final output of the decoder
            filter_len: int, length of the conv operatations
            num_layers: int, number of conv stages, each one doubles the dilation rate
            num_stages: int, after every num_stages, reset the dilation rate,
            final_conditioning: bool, if true conditions the output of the res blocks
            final_activation : str, name of the final activation function
    '''
    
    sig_shift = TemporalShift(shift=1,name=base_name+'temporal_shift')(signal)
    _,sig_len,_ = signal._keras_shape
    
    sig = Conv1D(filters=width,
                   kernel_size=filt_len,
                   padding='causal',
                   name=base_name+"convstart")(sig_shift)

    for idx in range(num_layers):
        dilation = 2**(idx % num_stages)
        sig = noskip_block(sig,width,filt_len,dilation=dilation,
                                 base_name=base_name+'noskip_block'+str(idx)+'_dil'+str(dilation),
                                 encoding=encoding,
                                 )


    if final_conditioning:
        sig_conv = Conv1D(filters=width,
               kernel_size=1,
               padding='causal',
               name=base_name+"sig_conv")(sig)
        conv_encoding = Conv1D(filters=width,
                       kernel_size=1,
                       padding='causal',
                       name=base_name+"conv_encoding")(encoding)
        sig_conditioned = AddEncoder()([sig_conv,conv_encoding])
        sig_conditioned = Activation('relu')(sig_conditioned)
        sig_out = Conv1D(filters=out_width,
                       kernel_size=1,
                       padding='causal',
                       name=base_name+"decoder_out_pre_act")(sig_conditioned)
    else:
        sig_out = Conv1D(filters=out_width,
               kernel_size=1,
               padding='causal',
               name=base_name+"decoder_out_pre_act")(sig)
    decoder_out = Activation(final_activation,name=
                         base_name+'decoder_out')(sig_out)
    return decoder_out


