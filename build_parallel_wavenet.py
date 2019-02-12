from tensorflow.python.debug.lib.debug_data import InconvertibleTensorProto
from tensorflow.python import debug as tf_debug

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
from distutils.util import strtobool
fs = os.path.sep

def my_filter(datum, tensor):
  """Only checks for nan vals
  """

  _ = datum  # Datum metadata is unused in this predicate.

  if isinstance(tensor, InconvertibleTensorProto):
    # Uninitialized tensor doesn't have bad numerical values.
    # Also return False for data types that cannot be represented as numpy
    # arrays.
    return False
  elif (np.issubdtype(tensor.dtype, np.floating) or
#        np.issubdtype(tensor.dtype, np.complex) or
        np.issubdtype(tensor.dtype, np.integer)):
    return  np.any(np.isnan(tensor)) and len(tensor) == 8
  else:
    return False

class TrainSequence(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x,batch_y

def get_default_args(func):
    sig = signature(func)
    return {k: v.default
            for k, v in sig.parameters.items()
            if v.default is not Parameter.empty }
    


def build_parallel_wavenet(signal_shape,encoding_shape,
                           teacher_path,teacher_processor,teacher_processor_kwargs,
                           stft_scale,
                           width=64,skip_width=64,out_width=64,
                           layer_list=None,num_stages=10,
                           teacher_preprocess_func_str=None,
                           num_student_samples=200,
                           log_scale_min=-5.0,log_scale_max=5.0,
                           quantize_output=True,
                           remove_dist_limits=True,
                           power_loss_scale=1,
                           frame_length=1024,frame_step=256,
                           ):
    #teacher set up (do this first in order to avoid possible name conflicts)
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
                                            log_scale_min=log_scale_min,
                                            log_scale_max=log_scale_max,
                                            base_name="iaf_"+str(flow_idx))
        shift_total = shift + shift_total*scale
        scale_total *= scale
        log_scale_total +=log_scale
        
#    student_output = Lambda(lambda x : tf.floor(x))(x)
    student_output =x
    student_model = keras.models.Model([signal_layer,encoding_input],student_output)
    
    

    
    def model_loss(avg_power, model_output):
        
        
        iaf_dist = tfp.distributions.Logistic(loc=shift_total,scale=scale_total)
        
        student_entropy = K.sum(iaf_dist.entropy(),axis=(1,2))
        
        transformed_sample = iaf_dist.sample(num_student_samples)
        transformed_sample = tf.reshape(transformed_sample,
                                        shape=[num_student_samples*nbatches,sig_len,1])
        

        
        #power loss
        model_output_scaled = model_output/(stft_scale)
        model_stft = tf.contrib.signal.stft(model_output_scaled[...,0],
                                 frame_length=frame_length,
                                 frame_step=frame_step)
        model_avg_pow = tf.reduce_mean(tf.pow(tf.abs(model_stft),2),axis=1)
        model_avg_pow = tf.expand_dims(model_avg_pow,axis=-1) #just to make keras happy
        power_loss = tf.reduce_sum(tf.square(avg_power-model_avg_pow),axis=(1,2))
        
        #teacher model
        
        preproc_student_output = teacher_preprocessor(model_output)

        teacher_output = teacher_model([encoding_input,preproc_student_output])
        tiled_teacher_output = tf.tile(teacher_output,[num_student_samples,1,1])
        
        output_channels = teacher_output._keras_shape[-1]
        teacher_dist = get_output_processor(teacher_processor,
                                        output_channels,
                                        teacher_processor_kwargs)
        
        if quantize_output:
            transformed_sample = tf.ceil(transformed_sample-0.5)
            teacher_dist.quantize=True
        else:
            teacher_dist.quantize=False
        #remove distribution limits
        fix_y_true=True
        if remove_dist_limits:
            if hasattr(teacher_dist,'low'):
                teacher_dist.low = None
            if hasattr(teacher_dist,'high'):
                teacher_dist.high = None
            fix_y_true=False

        teacher_logprob = teacher_dist.loss(transformed_sample,
                                            tiled_teacher_output,
                                            fix_y_true=fix_y_true)
        teacher_logprob = tf.reshape(teacher_logprob,
                                     shape=[num_student_samples,nbatches])
        avg_teacher_log_prob = tf.reduce_mean(teacher_logprob,axis=0)

#        return K.mean(-student_entropy+avg_teacher_log_prob)
        return K.mean(power_loss_scale*power_loss-student_entropy+avg_teacher_log_prob)
    
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
parser.add_argument('--config_json', dest='config_json',
            action='store', default=None,
            help='Path to the student config json')
parser.add_argument('--debug', dest='debug',
            action='store', default=False,type=strtobool,
            help='if true call tf_debug')
args = parser.parse_args()

if args.debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
    sess.add_tensor_filter('my_filter', my_filter)
    K.set_session(sess)
'''Defaults
'''
model_dict = get_default_args(build_parallel_wavenet)
train_dict = {'batch_size':8,'patience':10,'epochs':250}
if args.config_json is not None:
    with open(args.config_json,'r') as f:
        config_json = json.load(f)
        model_dict.update(config_json['model_dict'])
        train_dict.update(config_json['train_dict'])
        
batch_size = train_dict['batch_size']
patience = train_dict['patience']
epochs = train_dict['epochs']


with open(args.teacher_config_json,'r') as f:
    teacher_config = json.load(f)
teacher_generator_config = teacher_config['generator_dict']
teacher_model_config = teacher_config['model_dict']


model_dict['teacher_path'] = args.teacher_model
model_dict['teacher_processor'] = teacher_model_config['output_processor']
model_dict['teacher_processor_kwargs'] = teacher_model_config['output_processor_kwargs']
model_dict['teacher_preprocess_func_str'] = teacher_model_config['preprocess_func_str']

sig_len = teacher_generator_config['expected_len']
model_dict['signal_shape'] = [batch_size,sig_len,1]



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
    
assert test_data['stft_scale'] == train_data['stft_scale']
model_dict['stft_scale']=int(train_data['stft_scale'])
model_dict['frame_length']=int(train_data['frame_length'])
model_dict['frame_step']=int(train_data['frame_step'])
#need to set ndim of powers to 3 to make keras happy
train_powers = np.expand_dims(train_powers,axis=-1)
test_powers = np.expand_dims(test_powers,axis=-1)

model_dict['encoding_shape'] = [batch_size,train_encodings.shape[1],train_encodings.shape[2]]


train_seq = TrainSequence(train_encodings,train_powers,batch_size=batch_size)
test_seq = TrainSequence(test_encodings,test_powers,batch_size=batch_size)
#sys.exit()



#Model set up
all_dict = {'model_dict':model_dict,'train_dict':train_dict}
with open(args.save_path+'_options.json','w') as f:
    json.dump(all_dict,f,indent=4)
print('model setup')
model,model_loss = build_parallel_wavenet(**model_dict
                             )



model.compile(optimizer=AdamWithWeightnorm(),
          loss=model_loss)

#sys.exit()
####=Normal Training
csv_log = keras.callbacks.CSVLogger(args.save_path+'.csv')
lr_plateau = keras.callbacks.ReduceLROnPlateau(factor=0.2,patience=7)
early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',
                                        patience=patience,
                                        verbose=0, mode='auto')
model_checkpoint = keras.callbacks.ModelCheckpoint(args.save_path+'.hdf5',
                                                   save_best_only=True)
model.fit(x=train_encodings,y=train_powers,
          validation_data=(test_encodings,test_powers),
          batch_size=batch_size,
          epochs=epochs,
        verbose=1,
        callbacks=[early_stop,model_checkpoint,csv_log,lr_plateau],)

###NAN Finding
#csv_log = keras.callbacks.CSVLogger(args.save_path+'.csv')
#model_checkpoint = keras.callbacks.ModelCheckpoint(args.save_path+'.hdf5',
#                                                   save_best_only=True,
#                                                   monitor='loss')
#
#model.fit_generator(train_seq,steps_per_epoch=1,
#          epochs=epochs,
#        verbose=1,
#        callbacks=[model_checkpoint,csv_log],)






