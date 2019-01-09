import argparse
import keras
import keras.backend as K
from scipy.io.wavfile import read
from keras_wavenet.utils.wavenet_utils import (simple_load_wavfile,
                                           sample_to_categorical,
                                           categorical_to_sample)
from keras_wavenet.layers.wavenet import custom_objs
from keras_wavenet.utils.queued_wavenet_utils import load_model,batch_model_reset_states,model_output_transform
from keras_wavenet.utils.wavenet_utils import inv_mu_law_numpy
import pickle
import numpy as np
import sys
import librosa
import os
import json
fs = os.path.sep

'''Converts and tests the wavenet model using the queued method as described 
    here : https://arxiv.org/pdf/1611.09482.pdf
    It is much more efficient over the naive method.
'''
def synthesize(model,encoding,num_timesteps,mode='no_transform',
               init_state=None,verbose=False,):
        
    batch_model_reset_states(model)

    num_ch = 1
    num_batch,encoding_len,_ = encoding.shape
    full_audio = np.zeros((num_batch,num_timesteps,num_ch),dtype=np.float32)
    
    encoding_hop = num_timesteps // encoding_len
    if init_state is None:
        y = np.zeros((num_batch,1,num_ch),dtype=np.float32)
    else:
        y = init_state
    for idx in range(num_timesteps):
        if idx % 100 == 0 and verbose:
            sys.stdout.write("\r "+str(idx)+" / " +str(num_timesteps))
            sys.stdout.flush()

        enc_idx = idx // encoding_hop
        enc = np.expand_dims(encoding[:,enc_idx,:],axis=1)
        model_out = model.predict([enc,y])

        model_out = model_output_transform(model_out,mode)-128
        y = model_out / 128.
        full_audio[:,idx,:] = inv_mu_law_numpy(model_out[:,0,:])
        
    return full_audio
parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model',
                action='store', required=True,
                help='path to the model hdf5')
parser.add_argument('--generator', dest='generator',
                action='store', required=True,
                help='path to the generator')
parser.add_argument('--folder', dest='folder',
                action='store',default=None,
                help='path to the wavfile to encode')
parser.add_argument('--output_folder', dest='output_folder',
                action='store',default=None,
                help='path write output samples')
parser.add_argument('--config_json', dest='config_json',
                action='store',default=None,
                help='path to the config json')
args = parser.parse_args()
output_folder = args.output_folder if args.output_folder.endswith(fs) else args.output_folder+fs
os.makedirs(output_folder,exist_ok=True)

config_json = json.load(open(args.config_json,'r'))
sample_rate = config_json['generator_dict']['load_kwargs']['sample_rate']
num_timesteps = config_json['generator_dict']['expected_len']
encoding_size = config_json['model_dict']['latent_size']
preprocess_func_str = config_json['model_dict']['preprocess_func_str']

batch_size = 32

generator = pickle.load(open(args.generator,'rb'))
generator.random_transforms = False
train_gen = generator.flow_from_directory(args.folder,
                                              shuffle=True,
                                              follow_links=True,
                                              batch_size=32,
                                              )
test_x,test_y,filenames = train_gen.next(return_filenames=True)
encoder = load_model(args.model,queued=False,new_outputs=['z_mean','z_log_var'],custom_objects=custom_objs)

encoding_1,encoding_var_1 = encoder.predict(test_x)

model = load_model(args.model,queued=True,
                   new_inputs=['latent_sample','temporal_shift'],
                            batch_input_shapes=[(None,1,encoding_size),(None,1,1)],
                            custom_objects=custom_objs,batch_size=batch_size)


print('synthesizing')
full_audio = synthesize(model,encoding_1,num_timesteps,mode='sampling',verbose=True,
                        preprocess_func_str=preprocess_func_str)

for idx,waveform in enumerate(full_audio):
    librosa.output.write_wav(output_folder+'test_'+str(idx)+'.wav',waveform,sr=sample_rate)