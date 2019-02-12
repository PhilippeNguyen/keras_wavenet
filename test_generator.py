
import argparse
from keras_wavenet.utils.wavenet_utils import (simple_load_wavfile,
                                           sample_to_categorical,
                                           categorical_to_sample)
from keras_wavenet.layers.wavenet import custom_objs
from keras_wavenet.utils.queued_wavenet_utils import load_model
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
fs = os.path.sep


parser = argparse.ArgumentParser()
parser.add_argument('--folder', dest='folder',
                action='store',default=None,
                help='path to the wavfile to encode')
parser.add_argument('--output_folder', dest='output_folder',
                action='store',default=None,
                help='path write output samples')
parser.add_argument('--config_json', dest='config_json',
                action='store',default=None,
                help='path to the config json')
parser.add_argument('--num_timesteps', dest='num_timesteps',
                action='store',default=None,type=int,
                help='number of timesteps to generate,must be a multiple of the encoding_len')

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
output_processor_kwargs = config_json['model_dict']['output_processor_kwargs']
if args.num_timesteps is None:
    num_timesteps = config_json['generator_dict']['expected_len']
else:
    num_timesteps = args.num_timesteps
    config_json['generator_dict']['expected_len'] = num_timesteps
    config_json['generator_dict']['target_size'] = [num_timesteps,1]

batch_size = 8

generator = WavGenerator(**config_json['generator_dict'])
generator.random_transforms = False
train_gen = generator.flow_from_directory(args.folder,
                                              shuffle=True,
                                              follow_links=True,
                                              batch_size=batch_size,
                                              )
test_x,test_y,filenames = train_gen.next(return_filenames=True)
if used_mu_law:
    test_x = inv_mu_law_numpy(test_x)
for idx,waveform in enumerate(test_x):
    librosa.output.write_wav(output_folder+'test_'+str(idx)+'.wav',waveform,sr=sample_rate)