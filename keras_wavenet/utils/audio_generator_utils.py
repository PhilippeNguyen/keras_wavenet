import keras.backend as K
import numpy as np
from scipy.io import wavfile
import os
from keras.preprocessing.image import Iterator
import librosa
from scipy.signal import resample_poly
from keras_wavenet.utils.wavenet_utils import sample_to_categorical,mu_law
import random

def set_channel(sequence,num_channels,single_channel_idx=0):
    '''Converts a sequence of (num_timesteps,input_num_channels)
        or (num_timesteps,) to (num_timesetps,num_channels).
        If num_channel == 1, then single_channel_idx chooses the idx for 
        generating the single channel audio from a two-channel sequence
    '''
    if sequence.ndim == 1:
        sequence= np.expand_dims(sequence,axis=1)
        
        if num_channels == 2:
            sequence = np.repeat(sequence,2,axis=1)
    elif sequence.ndim == 2:
        if num_channels == 1:
            sequence = np.expand_dims(sequence[:,single_channel_idx],axis=-1)
    else:
        raise Exception("only num_channel of 1 or 2 works")
        
    return sequence


def normalize_wav(wav_data):
    if wav_data.dtype.name == 'float32':
        pass
    elif wav_data.dtype.name == 'int16':
        wav_data = wav_data*3.0517578125e-05
    elif wav_data.dtype.name == 'uint8':
        wav_data = (wav_data-127.5)/255
    else:
        raise ValueError("Unknown wav format : ",wav_data.dtype.name)
    return wav_data


def fix_seq_len(sequence,expected_len,
                       padding = 'post',
                       truncate = 'post',
                       val= 0.):
    '''Assumes sequence is (num_timesteps,num_features)
        also assumes sequence is a numpy array
    '''
    seq_len,num_features = np.shape(sequence)

    
    if seq_len < expected_len:
        output_seq = np.ones((expected_len,num_features))*val
        if padding == 'post':
            output_seq[:seq_len] = sequence
        elif padding == 'pre':
            output_seq[-seq_len:] = sequence
        else:
            print('padding',padding)
            raise Exception('padding arg not understood')
    
    elif seq_len > expected_len:
        if truncate == 'post':
            output_seq = sequence[:expected_len]
        elif truncate == 'pre':
            output_seq = sequence[-expected_len:]
        elif truncate == 'random':
            add_len = seq_len - expected_len
            start = int(np.random.uniform(0,add_len))
            output_seq =sequence[start:start+expected_len]
        else:
            print('truncate',truncate)
            raise Exception('truncate arg not understood')
            
    else:
        output_seq = sequence
    return output_seq


def load_wav(wav_file,
             sample_rate=44100,
             num_channels =2,
             single_channel_idx=0,
             dtype='float32',
             ):
    '''loads wav file as numpy array
    '''
    
    wav_rate,wav = wavfile.read(wav_file)
    wav = normalize_wav(wav)
    wav = set_channel(wav,
                      num_channels,
                      single_channel_idx)
    
    if wav_rate !=sample_rate:
         wav = resample_poly(wav,sample_rate,wav_rate)

        
    return wav.astype(dtype)


class WavGenerator(object):
    '''Generator to be used for keras fit_generator
    '''
    def __init__(self,load_kwargs,target_size,expected_len=None,
                 noise_std=None,
                 volume_std=None,
                 load_method='librosa',
                 expand_dim=None,
                 abs_one_rescale=True,
                 mu_law=False,
                 random_transforms =True,
                 max_loc=None,
                 front_padding=0.0,
                 random_offset=False,
                 uint16_rescale=False
                 
                 ):
        self.load_kwargs = load_kwargs
        self.target_size = target_size
        self.noise_std = noise_std
        self.volume_std = volume_std
        self.load_method = load_method
        self.expand_dim = expand_dim
        self.abs_one_rescale =abs_one_rescale
        self.mu_law = mu_law
        self.random_transforms=random_transforms
        self.expected_len = expected_len
        self.max_loc =max_loc
        self.front_padding = front_padding
        self.random_offset = random_offset
        self.uint16_rescale = uint16_rescale
        
    def flow_from_directory(self,
                             directory,
                             batch_size=32,
                             shuffle=True,
                             seed=None,
                             data_format=None,
                             follow_links=False):
        return DirectoryIterator(directory,
                                 self,
                                 target_size=self.target_size,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 seed=seed,
                                 data_format=data_format,
                                 follow_links=follow_links)
        
    def time_transform(self,x):
        if self.random_transforms:
            if self.noise_std is not None and self.noise_std != 0:
                    x+= np.random.randn(*x.shape)*self.noise_std
            if self.volume_std is not None and self.volume_std !=0:
                x *= (1+np.random.randn()*self.volume_std)
            
            if self.random_offset:
                len_diff =  len(x) - self.expected_len 
                if len_diff > 0:
                    start = int(len_diff*random.random())
                    x = x[start:start+self.expected_len]
                    
        if self.front_padding > 0.0:
            x = np.pad(x,((self.front_padding,0),(0,0)),mode='constant')
            
        if self.max_loc is not None:
            zeros = np.zeros_like(x)
            max_ind = np.unravel_index(np.argmax(np.abs(x)),x.shape)[0]
            diff = max_ind - self.max_loc
            if diff > 0:
                zeros[:-diff] = x[diff:]
            if diff < 0:
                diff = -diff
                zeros[diff:] = x[:-diff]
            x = zeros
            
        if self.expand_dim is not None:
            x = np.expand_dims(x,axis=self.expand_dim)
            
        if self.expected_len is not None:
            expected_len = int(self.expected_len)
            x = fix_seq_len(x,expected_len)
        return x
    
    def preprocess_pipeline(self,x):
        if self.abs_one_rescale:
            x_min,x_max = (np.minimum(np.min(x),-1),np.maximum(np.max(x),1))
            x = 2*((x - x_min)/(x_max-x_min)) - 1
        if self.uint16_rescale:
            x_min,x_max = (np.minimum(np.min(x),-1),np.maximum(np.max(x),1))
            x = ((x - x_min)/(x_max-x_min))*(2**16-1)
            x = np.round(x)
        if self.mu_law:
            x = mu_law(x)

        return x
    
    def load_wavfile(self,file):
        if self.load_method == 'librosa':
            x,_ = librosa.load(file,**self.load_kwargs)
        elif self.load_method == 'scipy':
            x = load_wav(file,**self.load_kwargs)[:,0]
        return x
    
    def process_wavfile(self,file):
        x = self.load_wavfile(file)
        x = self.time_transform(x)
        x = self.preprocess_pipeline(x)

        return x

        
class DirectoryIterator(Iterator):
    '''Should function like the Keras Directory Iterator
    '''
    def __init__(self,
                 directory,
                 wav_data_generator,
                 target_size,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 follow_links=False):
        self.directory = directory
        self.target_size = tuple(target_size)
        self.wav_data_generator = wav_data_generator
        

        wav_format = {'.wav','.mp3'}
        
        self.filenames = []
        for root,folders,files in os.walk(directory,followlinks=True):
            for file in files:
                if os.path.splitext(file)[-1] in wav_format:
                    self.filenames.append(os.path.join(root,file))
        
        self.samples = len(self.filenames)


        print('Found %d files ' %
              (self.samples))

        super(DirectoryIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)


        
    def _get_batches_of_transformed_samples(self, index_array,return_filenames=False):
        batch_x = np.zeros(
            (len(index_array),) + self.target_size,
            dtype=K.floatx())
        batch_filenames = [0]*len(index_array)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            batch_filenames[i] = fname
            try:
                x = self.wav_data_generator.process_wavfile(os.path.join(self.directory, fname),
                                                            )
                batch_x[i] = x
            except Exception as e:
                print('error at ',fname)
                print(e)
                print('Setting as zeros')
            

        batch_y = batch_x.copy()

        if return_filenames:
            return batch_x, batch_y, batch_filenames
        else:
            return batch_x, batch_y

    def next(self,return_filenames=False):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array,return_filenames)