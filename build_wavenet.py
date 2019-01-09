import tensorflow as tf
import keras
import argparse
from keras_wavenet.utils.audio_generator_utils import WavGenerator
import numpy as np
from keras.optimizers import Adam

import keras.backend as K
import pickle

from keras.layers import (Lambda,Reshape,Conv1D,Add,Activation,
                          Concatenate,Dense,RepeatVector,GRU,Bidirectional,
                          Multiply,MaxPool1D,Flatten,GlobalMaxPool1D,TimeDistributed,
                          Dropout,ZeroPadding1D,GaussianDropout)
from keras_wavenet.models.wavenet import build_wavenet_decoder,build_wavenet_encoder
from keras.losses import sparse_categorical_crossentropy
import os
import json
fs = os.path.sep
bn_axis = 2

def build_model(input_shape,dec_width=512,dec_skip_width=256,
                enc_width=512,
                num_en_layers=8,num_en_stages=2,enc_pool_size=None,
                num_dec_layers=30,num_dec_stages=10,
                latent_size=8,filter_len=3,epsilon_std=1.0,
                final_conditioning=True,
                final_activation='softmax',
                output_channels=257,
                preprocess_func_str = "lambda x : x/128."
): 
    
    num_steps,input_channels = input_shape
    input_layer = keras.layers.Input(shape = input_shape) 
    preprocess_func = eval(preprocess_func_str)
    scaled_input = Lambda(preprocess_func,output_shape=input_shape,
                          name='preprocessor')(input_layer)
    #Encoder
    encoding = build_wavenet_encoder(scaled_input,width=enc_width,filter_len=filter_len,
                                     num_layers=num_en_layers,num_stages=num_en_stages
                                     )
    encoding = Activation('relu')(encoding)
    
    if enc_pool_size is None:
        en = GlobalMaxPool1D()(encoding)
        en = Reshape((1,enc_width))(en)
    else:
        assert input_shape[0] % enc_pool_size == 0 
        en = MaxPool1D(pool_size=enc_pool_size)(encoding)
    
    #Latent Sampling
    z_mean = Conv1D(latent_size,kernel_size=1,name='z_mean')(en)
    z_log_var = Conv1D(latent_size,kernel_size=1,name='z_log_var',
                      )(en)
    _,enc_timesteps,_ = z_mean._keras_shape
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], enc_timesteps, latent_size),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var/2) * epsilon
    
    z = Lambda(sampling, output_shape=(enc_timesteps,latent_size,),
               name='latent_sample')([z_mean, z_log_var])
    
    #Decoder
    decoder_out = build_wavenet_decoder(z,scaled_input,
                                    width=dec_width,skip_width=dec_skip_width,
                                    out_width=output_channels,
                                    num_layers=num_dec_layers,num_stages=num_dec_stages,
                                    final_conditioning=final_conditioning,
                                    final_activation=final_activation
                                    )
    
    def vae_loss(y_true, model_output):
        y_true = K.cast(K.batch_flatten(y_true), tf.int32) + 128
        mse_loss = K.sum(sparse_categorical_crossentropy(y_true,model_output),axis=-1)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=(1,2))
        return K.mean(mse_loss + kl_loss )
    
    model = keras.models.Model(input_layer, decoder_out)
    
    return model,vae_loss





parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', dest='train_folder',
                action='store', required=True,
                help='train folder to load')
parser.add_argument('--valid_folder', dest='valid_folder',
            action='store', required=True,
            help='validation folder to load')
parser.add_argument('--save_path', dest='save_path',
        action='store', required=True,
        help='file name to save the model as')
parser.add_argument('--config_json', dest='config_json',
            action='store', default=None,
            help='Path to the config json')
args = parser.parse_args()

'''Defaults
'''
if args.config_json is None:
    generator_dict = {
    'target_size' : (6000,1),
    'expand_dim' : 1,
    'abs_one_rescale':True,
    'sample_to_categorical':False,
    'mu_law':True,
    'load_kwargs':{'sample_rate':20000,
                   'num_channels':1},
    'load_method':'scipy',
    'expected_len':int(6000),
    }
    model_dict={
    'enc_width':128,
    'dec_width':128,
    'dec_skip_width':100,
    'enc_pool_size':1000,
    'latent_size':32,
    'final_conditioning':True,
    'num_dec_layers':30,
    'num_dec_stages':10,
    'final_activation':'softmax',
    }
else:
    config_json = json.load(open(args.config_json,'r'))
    generator_dict = config_json['generator_dict']
    model_dict = config_json['model_dict']
    
batch_size = 8
patience = 10
epochs = 250

all_dict = {'generator_dict':generator_dict,'model_dict':model_dict}
json.dump(all_dict,open(args.save_path+'_options.json','w'),indent=4)

train_generator = WavGenerator(**generator_dict
                                        )
test_generator = WavGenerator(**generator_dict
                                        )
pickle.dump(test_generator,open(args.save_path+'_generator.pkl','wb'))


print('loading data')
train_gen = train_generator.flow_from_directory(args.train_folder,
                                              shuffle=True,
                                              follow_links=True,
                                              batch_size=batch_size)
valid_gen = test_generator.flow_from_directory(args.valid_folder,
                                          shuffle=True,
                                          follow_links=True,
                                          batch_size=batch_size)

test_x,test_y = train_gen.next()
#sys.exit()
train_gen.reset()
input_shape = np.shape(test_x)[1:]

num_train_samples =  train_gen.samples
num_train_steps_per_epoch = np.ceil(num_train_samples/batch_size)

num_valid_samples =  valid_gen.samples
num_valid_steps_per_epoch = np.ceil(num_valid_samples/batch_size)

#Model set up

print('model setup')
model,vae_loss = build_model(input_shape,
                             **model_dict
                             )



model.compile(optimizer=Adam(),
          loss=vae_loss)

#sys.exit()

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',
                                        patience=patience,
                                        verbose=0, mode='auto')
csv_log = keras.callbacks.CSVLogger(args.save_path+'.csv')
model_checkpoint = keras.callbacks.ModelCheckpoint(args.save_path+'.hdf5',
                                                   save_best_only=True)
model.fit_generator(train_gen,
                    steps_per_epoch=num_train_steps_per_epoch,
                    validation_data=valid_gen,
                    validation_steps=num_valid_steps_per_epoch,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[early_stop,model_checkpoint,csv_log],

                    )





