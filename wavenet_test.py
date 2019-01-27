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
import pickle
import numpy as np
import sys
import librosa
import os
import json
from functools import partial
fs = os.path.sep


parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model',
                action='store', required=True,
                help='path to the model hdf5')
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
used_mu_law = config_json['generator_dict']['mu_law']
output_processor = config_json['model_dict']['output_processor']
batch_size = 32

generator = WavGenerator(**config_json['generator_dict'])
generator.random_transforms = False
train_gen = generator.flow_from_directory(args.folder,
                                              shuffle=True,
                                              follow_links=True,
                                              batch_size=32,
                                              )
test_x,test_y,filenames = train_gen.next(return_filenames=True)



#need to handle stochastic encoding
model = load_model(args.model,compile=False,
                     custom_objects=custom_objs)


output_channels = model.output_shape[-1]
if output_processor == 'sparse_categorical':
    from keras_wavenet.models.audio_outputs import SparseCategorical
    sampler = SparseCategorical(num_classes=output_channels,bias=128).sample
elif output_processor.lower() == 'dmll_tfp':
    from keras_wavenet.models.audio_output_tf import DMLL_tfp
    sampler = partial(DMLL_tfp(num_classes=output_channels//3).sample,model=model)
model_out = model.predict(test_x)
out = sampler(model_out)