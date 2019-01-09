
import keras
from keras.engine.saving import model_from_config,preprocess_weights_for_loading
from keras.utils.io_utils import h5dict
from keras.utils.generic_utils import to_list
import json
import h5py

def get_model_config(filepath):

    opened_new_file = not isinstance(filepath, h5py.Group)
    f = h5dict(filepath, 'r')
    try:
        model_config = f['model_config']
        if model_config is None:
            raise ValueError('No model found in config.')
        model_config = json.loads(model_config.decode('utf-8'))
    finally:
        if opened_new_file:
            f.close()
    return model_config