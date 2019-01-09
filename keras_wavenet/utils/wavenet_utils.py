import numpy as np
from keras.utils import to_categorical
from scipy.io.wavfile import read
from scipy.signal import resample_poly
    

def mu_law(x, mu=255, dtype='float32'):
  out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
  out = np.floor(out * 128)
  out = out.astype(dtype=dtype)
  return out

def inv_mu_law_numpy(x, mu=255.0):

    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out


def sample_to_categorical(x,bias=128,num_classes=256):
    '''converts a numpy array from wavfile samples to categorical array as described 
        in wavenet paper (no mu_law transformation)
    '''
    x_categorical = to_categorical(x,num_classes=num_classes)
    return x_categorical
    
def categorical_to_sample(x,bias=128):
    '''converts back from categorical array to waveform (no mu_law transformation)
    '''
    out = np.argmax(x,axis=-1)-bias
    return out

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

def simple_load_wavfile(path,sample_rate,duration):
    '''Simple method to load a wavfile of fixed length/sample-rate
    
        Personally, I use a much more complicated set of code for reading
        audio files, but it is a monstrosity; don't want it here.
        
        Args:
            path: path to the wavfile
            sample_rate : sample rate, will resample the audio to this rate
            duration: int, in seconds, will either crop or padd with zeros 
    '''
    expected_len = int(duration*sample_rate)
    og_sr,x = read(path)
    x = normalize_wav(x)
    
    if len(x.shape) == 2:
        #use only one channel if wavfile is multichannel
        x = x[...,0]
    x = resample_poly(x,sample_rate,og_sr)
    
    if len(x) > expected_len:
        x = x[:expected_len]
    else:
        zeros_x = np.zeros((expected_len))
        zeros_x[:len(x)] = x
        x = zeros_x
    
    return x
