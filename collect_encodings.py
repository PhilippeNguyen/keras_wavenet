import argparse
from keras_wavenet.layers.wavenet import custom_objs
from keras_wavenet.utils.queued_wavenet_utils import load_model,batch_model_reset_states
from keras_wavenet.utils.audio_generator_utils import WavGenerator
import numpy as np
import sys
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
parser.add_argument('--output_file', dest='output_file',
                action='store',default=None,
                help='name of the output filename (.npy)')
parser.add_argument('--config_json', dest='config_json',
                action='store',default=None,
                help='path to the config json')
parser.add_argument('--num_timesteps', dest='num_timesteps',
                action='store',default=None,type=int,
                help='number of timesteps to generate,must be a multiple of the encoding_len')

args = parser.parse_args()
output = args.output_file if args.output_file.endswith('.npy') else args.output_file+'.npy'

config_json = json.load(open(args.config_json,'r'))
sample_rate = config_json['generator_dict']['load_kwargs']['sample_rate']
num_timesteps = config_json['generator_dict']['expected_len']
encoding_size = config_json['model_dict']['latent_size']
preprocess_func_str = config_json['model_dict']['preprocess_func_str']
used_mu_law = config_json['generator_dict']['mu_law']
output_processor = config_json['model_dict']['output_processor']
output_processor_kwargs = config_json['model_dict']['output_processor_kwargs']
batch_size = 32

generator = WavGenerator(**config_json['generator_dict'])
generator.random_transforms = False
train_gen = generator.flow_from_directory(args.folder,
                                              shuffle=True,
                                              follow_links=True,
                                              batch_size=batch_size,
                                              )

#need to handle stochastic encoding
encoder = load_model(args.model,queued=False,new_outputs=['z_mean'],
                     custom_objects=custom_objs)


encoding_list = []
samples_remaining = train_gen.samples
while samples_remaining > 0:
    sys.stdout.write("\r {}".format(samples_remaining))
    sys.stdout.flush()
    test_x,test_y, = train_gen.next()
    encodings = encoder.predict(test_x)
    if samples_remaining < batch_size:
        encodings = encodings[:samples_remaining]
    encoding_list.append(encodings)
    samples_remaining -= encodings.shape[0]

encodings = np.concatenate(encoding_list)
np.save(output,encodings)