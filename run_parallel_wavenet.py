import argparse
import keras
import keras.backend as K
from scipy.io.wavfile import read
from keras_wavenet.utils.wavenet_utils import (simple_load_wavfile,
                                           sample_to_categorical,
                                           categorical_to_sample)
from keras_wavenet.layers.wavenet import custom_objs
from keras.models import load_model
from keras_wavenet.utils.wavenet_utils import inv_mu_law_numpy
from keras_wavenet.utils.audio_generator_utils import WavGenerator
from keras_wavenet.models.audio_outputs import get_output_processor
import pickle
import numpy as np
import sys
import librosa
import os
import json
from functools import partial
import tensorflow_probability as tfp
fs = os.path.sep

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model',
                action='store', required=True,
                help='path to the model hdf5')
#parser.add_argument('--config_json', dest='config_json',
#                action='store',required=True,
#                help='path to the config json')
parser.add_argument('--npz', dest='npz',
                action='store', required=True,
                help='path to the encodings (.npz files)')
args = parser.parse_args()

data = np.load(args.npz)
encodings = data['encodings']

model = load_model(args.model,custom_objects=custom_objs,compile=False)

#with open(args.config_json,'r') as f:
#    config_json = json.load(f)

model_inputs = model.inputs
signal_tensor,encoding_tensor = model_inputs
sig_shape = signal_tensor.get_shape().as_list()
enc_shape = encoding_tensor.get_shape().as_list()
num_batches,sig_len,_ = sig_shape

base_dist = tfp.distributions.Logistic(loc=0,scale=1)
sampler = base_dist.sample(sig_shape)
#model_sample = model([sample,K.placeholder(shape=enc_shape)])
#
#out = model_sample.predict(encodings[:num_batches])

sample = sampler.eval(session=K.get_session())
out = model.predict([sample,encodings[:num_batches]])

#my_layer = model.get_layer('iaf_3__iaf_logscale_unclipped')
my_layer = model.get_layer('iaf_3_iaf_logscale')
bb = my_layer.output.eval(session=K.get_session(),feed_dict={model.inputs[0]:sample,model.inputs[1]:encodings[:num_batches]})
#my_layer = model.get_layer('iaf_2__iaf_logscale_unclipped')
#my_layer = model.get_layer('iaf_1__iaf_logscale_unclipped')
#my_layer = model.get_layer('iaf_0__iaf_logscale_unclipped')
#my_layer = model.get_layer('iaf_3__iaf_shift')
#my_layer = model.get_layer('iaf_0_decoder_out_pre_act')