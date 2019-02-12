import tensorflow as tf
import keras
import argparse
from keras_wavenet.utils.audio_generator_utils import WavGenerator
import numpy as np
from keras.optimizers import Adam
from keras_wavenet.weightnorm import AdamWithWeightnorm

import keras.backend as K
import pickle

from keras.layers import (Lambda,Reshape,Conv1D,Add,Activation,
                          Concatenate,Dense,RepeatVector,GRU,Bidirectional,
                          Multiply,MaxPool1D,Flatten,GlobalMaxPool1D,TimeDistributed,
                          Dropout,ZeroPadding1D,GaussianDropout)
from keras_wavenet.models.wavenet import build_wavenet_decoder,build_wavenet_encoder
from keras_wavenet.models.audio_outputs import get_output_processor
import sys
import os
import json
from inspect import signature,Parameter
fs = os.path.sep


def get_default_args(func):
    sig = signature(func)
    return {k: v.default
            for k, v in sig.parameters.items()
            if v.default is not Parameter.empty }
    

def build_model(input_shape,dec_width=512,dec_skip_width=256,
                enc_width=512,
                num_en_layers=8,num_en_stages=2,enc_pool_size=None,
                num_dec_layers=30,num_dec_stages=10,
                latent_size=8,filter_len=3,epsilon_std=1.0,
                final_conditioning=True,
                final_activation='softmax',
                output_channels=257,
                preprocess_func_str = "lambda x : x/128.",
                stochastic_encoding=False,
                output_processor='sparse_categorical',
                output_processor_kwargs = None,
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
    if stochastic_encoding:
        
        z_log_var = Conv1D(latent_size,kernel_size=1,
                           kernel_initializer=keras.initializers.constant(-0.1)
                          )(en)
        z_log_var = Lambda(lambda x: K.minimum(x,6.0),name='z_log_var',)(z_log_var)
        _,enc_timesteps,_ = z_mean._keras_shape
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], enc_timesteps, latent_size),
                                      mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_var/2) * epsilon
        
        z = Lambda(sampling, output_shape=(enc_timesteps,latent_size,),
                   name='decoder_input')([z_mean, z_log_var])
    else:
        z = Lambda(lambda x: x,name='decoder_input')(z_mean)
    
    #Decoder
    decoder_out = build_wavenet_decoder(scaled_input,z,
                                    width=dec_width,skip_width=dec_skip_width,
                                    out_width=output_channels,
                                    num_layers=num_dec_layers,num_stages=num_dec_stages,
                                    final_conditioning=final_conditioning,
                                    final_activation=final_activation
                                    )
    
    model = keras.models.Model(input_layer, decoder_out)
    
    #build loss
    px_loss_func = get_output_processor(output_processor,
                                        output_channels,
                                        output_processor_kwargs).loss
        
    def vae_loss(y_true, model_output):
        px_loss = px_loss_func(y_true,model_output)
        if stochastic_encoding:
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=(1,2))
            return K.mean(px_loss + kl_loss )
        return px_loss
    
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
generator_dict = get_default_args(WavGenerator)
model_dict = get_default_args(build_model)
train_dict = {'batch_size':8,'patience':10,'epochs':250}
if args.config_json is not None:
    config_json = json.load(open(args.config_json,'r'))
    generator_dict.update(config_json['generator_dict'])
    model_dict.update(config_json['model_dict'])
    train_dict.update(config_json['train_dict'])


all_dict = {'generator_dict':generator_dict,'model_dict':model_dict}
json.dump(all_dict,open(args.save_path+'_options.json','w'),indent=4)

train_generator = WavGenerator(**generator_dict
                                        )
test_generator = WavGenerator(**generator_dict,
                                        )
test_generator.random_transforms = False
pickle.dump(test_generator,open(args.save_path+'_generator.pkl','wb'))


print('loading data')
train_gen = train_generator.flow_from_directory(args.train_folder,
                                              shuffle=True,
                                              follow_links=True,
                                              batch_size=train_dict['batch_size'])
valid_gen = test_generator.flow_from_directory(args.valid_folder,
                                          shuffle=True,
                                          follow_links=True,
                                          batch_size=train_dict['batch_size'])

test_x,test_y,filenames = train_gen.next(return_filenames=True)
#sys.exit()
train_gen.reset()
input_shape = np.shape(test_x)[1:]

num_train_samples =  train_gen.samples
num_train_steps_per_epoch = np.ceil(num_train_samples/train_dict['batch_size'])

num_valid_samples =  valid_gen.samples
num_valid_steps_per_epoch = np.ceil(num_valid_samples/train_dict['batch_size'])

#Model set up

print('model setup')
model,vae_loss = build_model(input_shape,
                             **model_dict
                             )



model.compile(optimizer=AdamWithWeightnorm(),
          loss=vae_loss)

#sys.exit()

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',
                                        patience=train_dict['patience'],
                                        verbose=0, mode='auto')
csv_log = keras.callbacks.CSVLogger(args.save_path+'.csv')
model_checkpoint = keras.callbacks.ModelCheckpoint(args.save_path+'.hdf5',
                                                   save_best_only=True)
lr_plateu = keras.callbacks.ReduceLROnPlateau(factor=0.2,patience=7)
model.fit_generator(train_gen,
                    steps_per_epoch=num_train_steps_per_epoch,
                    validation_data=valid_gen,
                    validation_steps=num_valid_steps_per_epoch,
                    epochs=train_dict['epochs'],
                    verbose=1,
                    callbacks=[early_stop,model_checkpoint,csv_log,lr_plateu],

                    )





