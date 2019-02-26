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

parser.add_argument('--npz', dest='npz',
                action='store', required=True,
                help='path to the encodings (.npz files)')
args = parser.parse_args()

data = np.load(args.npz)
encodings = data['encodings']

model = load_model(args.model,custom_objects=custom_objs,compile=False)


model_inputs = model.inputs
signal_tensor,encoding_tensor = model_inputs
sig_shape = signal_tensor.get_shape().as_list()
enc_shape = encoding_tensor.get_shape().as_list()
num_batches,sig_len,_ = sig_shape

base_dist = tfp.distributions.Logistic(loc=0,scale=1)
sampler = base_dist.sample(sig_shape)


sample = sampler.eval(session=K.get_session())
out = model.predict([sample,encodings[:num_batches]])

