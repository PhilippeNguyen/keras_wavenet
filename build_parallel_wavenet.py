import tensorflow as tf
import keras
import argparse
from keras_wavenet.utils.audio_generator_utils import WavGenerator
import numpy as np
from keras.optimizers import Adam
from keras_wavenet.weightnorm import AdamWithWeightnorm
from keras_wavenet.models.parallel_wavenet import wavenet_iaf_step
from keras_wavenet.layers.wavenet import custom_objs
import keras.backend as K
from keras.models import load_model
from keras_wavenet.utils.queued_wavenet_utils import load_model as q_load_model
import pickle

from keras.layers import (Lambda,Reshape,Conv1D,Add,Activation,
                          Concatenate,Dense,RepeatVector,GRU,Bidirectional,
                          Multiply,MaxPool1D,Flatten,GlobalMaxPool1D,TimeDistributed,
                          Dropout,ZeroPadding1D,GaussianDropout)
from keras_wavenet.models.wavenet import build_wavenet_decoder,build_wavenet_encoder
from keras_wavenet.models.audio_outputs import get_output_processor
from keras_wavenet.models.audio_outputs_tf import Logistic_tfp

import sys
import os
import json
from inspect import signature,Parameter
import tensorflow_probability as tfp
fs = os.path.sep


def get_default_args(func):
    sig = signature(func)
    return {k: v.default
            for k, v in sig.parameters.items()
            if v.default is not Parameter.empty }

def build_parallel_wavenet(signal_shape,encoding_shape,
                           teacher_path,teacher_processor,teacher_processor_kwargs,
                           width=64,skip_width=64,out_width=64,
                           layer_list=None,num_stages=10,
                           teacher_preprocess_func_str=None,
                           stft_scale=1,
                           ):
    
    teacher_model = q_load_model(teacher_path,queued=False,custom_objects=custom_objs,
                                 new_inputs=['decoder_input','temporal_shift'],
                                 batch_input_shapes=[encoding_shape,signal_shape],
                               )
    if teacher_preprocess_func_str is not None:
        teacher_preprocess_func = eval(teacher_preprocess_func_str)
    else:
        teacher_preprocess_func = lambda x : x
    teacher_preprocessor = Lambda(teacher_preprocess_func,output_shape=signal_shape[1:],
                      name='teacher_preprocessor')
    
    nbatches,sig_len,_ =  signal_shape
    
    base_dist = tfp.distributions.Logistic(loc=0.,scale=1.0)
    signal = base_dist.sample(signal_shape)
    signal_layer = keras.layers.Input(tensor=signal)
    encoding_input = keras.layers.Input(batch_shape=encoding_shape)
    
    if layer_list is None:
        layer_list = [10,10,10,30]
        
    x = signal_layer
    #collect parameters for the output iaf distribution, we'll sample from it, not use x directly.
    shift_total= 0
    scale_total= 1
    log_scale_total = 0
    for flow_idx,layers in enumerate(layer_list):
        x,shift,scale,log_scale = wavenet_iaf_step(x,encoding_input,
                                            width,skip_width,out_width,
                                            num_layers=layer_list[flow_idx],
                                            num_stages=num_stages,
                                            base_name="iaf_"+str(flow_idx)+"_")
        shift_total = shift + shift_total*scale
        scale_total *= scale
        log_scale_total +=log_scale
        
    student_output = x
    student_model = keras.models.Model([signal_layer,encoding_input],student_output)
    

    
    def model_loss(avg_power, model_output):
        #power loss
        model_output_scaled = model_output/(stft_scale)
        model_stft = tf.contrib.signal.stft(model_output_scaled[...,0],
                                 frame_length=1024,
                                 frame_step=256)
        model_avg_pow = tf.reduce_mean(tf.pow(tf.abs(model_stft),2),axis=1)
        model_avg_pow = tf.expand_dims(model_avg_pow,axis=-1) #just to make keras happy
        power_loss = tf.reduce_sum(tf.square(avg_power-model_avg_pow),axis=(1,2))
        #teacher model
        
        iaf_dist = tfp.distributions.Logistic(loc=shift_total,scale=scale_total)
        
#        model_scale = model_output[...,0]
#        model_shift = model_output[...,1]
#        iaf_dist = tfp.distributions.Logistic(loc=model_shift,scale=model_scale)
        
#        iaf_output_processor = Logistic_tfp()._build(model_output)
#        iaf_dist = iaf_output_processor.dist
        
        transformed_sample = iaf_dist.sample()
        
        preproc_student_output = teacher_preprocessor(transformed_sample)

        teacher_output = teacher_model([encoding_input,preproc_student_output])
        
        
        output_channels = teacher_output._keras_shape[-1]
        teacher_dist = get_output_processor(teacher_processor,
                                        output_channels,
                                        teacher_processor_kwargs)
        teacher_dist._build(teacher_output)
        '''
        loss = teacher_dist.loss(transformed_sample,teacher_output)
#        loss = -K.sum(teacher_dist_tfp.log_prob(student_output[...,0]),axis=-1) #teacher-student cross entropy
        loss -= (K.sum(log_scale_total,axis=(1,2)) + 2*sig_len) #student entropy
        '''
        student_entropy = K.sum(iaf_dist.entropy(),axis=(1,2))
#        cross_entropy = K.sum(teacher_dist.dist.cross_entropy(iaf_dist),axis=-1)
#        kl_d = K.sum(teacher_dist.dist.kl_divergence(iaf_dist),axis=-1)
        teacher_logprob = teacher_dist.loss(model_output,teacher_output)
#        return K.mean(power_loss-student_entropy+cross_entropy)
#        return K.mean(power_loss+kl_d)
        return K.mean(0*power_loss-student_entropy+teacher_logprob)
    
    return student_model,model_loss




parser = argparse.ArgumentParser()
parser.add_argument('--train_npz', dest='train_encodings',
                action='store', required=True,
                help='path to the training encodings (.npz files)')
parser.add_argument('--valid_npz', dest='valid_encodings',
            action='store', required=True,
            help='path to the validation encodings (.npz files)')
parser.add_argument('--teacher_model', dest='teacher_model',
                action='store', required=True,
                help='path to the teacher model hdf5')
parser.add_argument('--teacher_config_json', dest='teacher_config_json',
            action='store', required=True,
            help='Path to the teacher config json')
parser.add_argument('--save_path', dest='save_path',
        action='store', required=True,
        help='file name to save the model as')
#parser.add_argument('--student_config_json', dest='student_config_json',
#            action='store', default=None,
#            help='Path to the student config json')
args = parser.parse_args()

'''Defaults
'''
#generator_dict = get_default_args(WavGenerator)
#model_dict = get_default_args(build_parallel_wavenet)
#if args.config_json is not None:
#    config_json = json.load(open(args.config_json,'r'))
#    generator_dict.update(config_json['generator_dict'])
#    model_dict.update(config_json['model_dict'])
batch_size = 8
patience = 10
epochs = 250

with open(args.teacher_config_json,'r') as f:
    teacher_config = json.load(f)
teacher_generator_config = teacher_config['generator_dict']
teacher_model_config = teacher_config['model_dict']
if teacher_generator_config['mu_law']:
    stft_scale = 128
else:
    stft_scale = 1
    
teacher_processor = teacher_model_config['output_processor']
teacher_processor_kwargs = teacher_model_config['output_processor_kwargs']
teacher_preprocess_func_str = teacher_model_config['preprocess_func_str']

sig_len = teacher_generator_config['expected_len']
sig_shape = [batch_size,sig_len,1]



train_data = np.load(args.train_encodings)
test_data = np.load(args.valid_encodings)
train_encodings,train_powers = (train_data['encodings'],train_data['powers'])
test_encodings,test_powers = (test_data['encodings'],test_data['powers'])

train_rem = (len(train_encodings)%batch_size)
if train_rem !=0:
    train_encodings = train_encodings[:-train_rem]
    train_powers = train_powers[:-train_rem]

test_rem = (len(test_encodings)%batch_size)
if test_rem !=0:
    test_encodings = test_encodings[:-test_rem]
    test_powers = test_powers[:-test_rem]

#need to set ndim of powers to 3 to make keras happy
train_powers = np.expand_dims(train_powers,axis=-1)
test_powers = np.expand_dims(test_powers,axis=-1)
    
enc_shape = [batch_size,train_encodings.shape[1],train_encodings.shape[2]]
#Model set up
#sys.exit()
print('model setup')
model,model_loss = build_parallel_wavenet(sig_shape,enc_shape,
                             teacher_path=args.teacher_model,
                             teacher_processor=teacher_processor,
                             teacher_processor_kwargs=teacher_processor_kwargs,
                             teacher_preprocess_func_str=teacher_preprocess_func_str,
                             stft_scale=stft_scale
                             )



model.compile(optimizer=AdamWithWeightnorm(),
          loss=model_loss)

#sys.exit()

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',
                                        patience=patience,
                                        verbose=0, mode='auto')
csv_log = keras.callbacks.CSVLogger(args.save_path+'.csv')
model_checkpoint = keras.callbacks.ModelCheckpoint(args.save_path+'.hdf5',
                                                   save_best_only=True)
lr_plateu = keras.callbacks.ReduceLROnPlateau(factor=0.2,patience=7)

model.fit(x=train_encodings,y=train_powers,
          validation_data=(test_encodings,test_powers),
          batch_size=batch_size,
          epochs=epochs,
        verbose=1,
        callbacks=[early_stop,model_checkpoint,csv_log,lr_plateu],)







