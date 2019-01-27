import argparse
from keras_wavenet.layers.wavenet import custom_objs
from keras_wavenet.utils.queued_wavenet_utils import load_model,batch_model_reset_states
from keras_wavenet.utils.audio_generator_utils import WavGenerator
import numpy as np
import sys
import os
import json
from functools import partial
import tensorflow as tf
import keras.backend as K

sess = K.get_session()
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
args = parser.parse_args()
output = args.output_file if args.output_file.endswith('.npz') else args.output_file+'.npz'

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
powers_list = []
tf_in = tf.placeholder(tf.float32,(None,num_timesteps,1))

'''Note that the stft is done on the audio output from the generator
    if the generator is in mu-law then the audio ranges from [-128,128],
    so the stft vals will be several orders of magnitude greater than audio
    with range [-1,1], 
    
    be sure to account for this.
'''
if used_mu_law:
    stft_scale = 128
else:
    stft_scale = 1
stft_fn = tf.contrib.signal.stft(tf_in[...,0]/stft_scale,
                                 frame_length=1024,
                                 frame_step=256)
avg_pow_fn = tf.reduce_mean(tf.pow(tf.abs(stft_fn),2),axis=1)

#stft_fn_scaled = tf.contrib.signal.stft(tf_in[...,0]/128,
#                                 frame_length=1024,
#                                 frame_step=256)
#avg_pow_fn_scaled = tf.reduce_mean(tf.pow(tf.abs(stft_fn_scaled),2),axis=1)
samples_remaining = train_gen.samples
while samples_remaining > 0:
    sys.stdout.write("\r {}".format(samples_remaining))
    sys.stdout.flush()
    test_x,test_y, = train_gen.next()
    encodings = encoder.predict(test_x)
    stft = stft_fn.eval(session=sess,feed_dict={tf_in:test_x})
    powers = avg_pow_fn.eval(session=sess,feed_dict={tf_in:test_x})
#    stft_scaled = stft_fn_scaled.eval(session=sess,feed_dict={tf_in:test_x})
#    powers_scaled = avg_pow_fn_scaled.eval(session=sess,feed_dict={tf_in:test_x})
    
    encoding_list.append(encodings)
    powers_list.append(powers)
    samples_remaining -= encodings.shape[0]

encodings = np.concatenate(encoding_list)
powers = np.concatenate(powers_list)
np.savez(output,encodings=encodings,powers=powers)
