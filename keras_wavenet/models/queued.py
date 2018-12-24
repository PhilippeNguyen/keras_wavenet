import keras
from keras.models import Model
import h5py
import json
from keras.engine.saving import model_from_config,preprocess_weights_for_loading
from keras.utils.io_utils import h5dict
from keras.utils.generic_utils import to_list
import keras.backend as K


def load_model(filepath,queued,batch_size=1,custom_objects=None,
               new_inputs=None,new_outputs=None,batch_input_shapes=None):
    '''
    
    Args:
        queued: bool, if true convert to queued model
    '''

    model = None
    opened_new_file = not isinstance(filepath, h5py.Group)
    f = h5dict(filepath, 'r')
    try:
        if queued:
            from ..layers.queued import QueuedConv1D
            if custom_objects is None:
                custom_objects = {}
            custom_objects.update({'QueuedConv1D':QueuedConv1D})
        model = _deserialize_model(f, batch_size=batch_size,custom_objects=custom_objects,
                                   new_inputs=new_inputs,new_outputs=new_outputs,
                                   batch_input_shapes=batch_input_shapes,
                                   convert_to_queued=queued)
    finally:
        if opened_new_file:
            f.close()
    return model



def convert_to_queued_config(config,batch_size=1):
    print('Converting all Conv1D layers to QueuedConv1D')
    new_config = config.copy()
    model_config = config['config']
    layers = model_config['layers']
    new_layers = []
    for layer in layers:
        new_layer = layer.copy()
        
        if layer['class_name'] == 'Conv1D':
            config = layer['config']
            for stride in config['strides']:
                if stride != 1:
                    raise ValueError('Conversion to QueuedConv1D requires all strides = 1')
            if config['padding'] != 'causal':
                raise ValueError('All convolutions must be causal for a queued model')
            new_layer['class_name'] = 'QueuedConv1D'
            new_layer['config']['batch_size'] = batch_size
                    
        new_layers.append(new_layer)
        
    model_config['layers'] = new_layers
    return new_config

def layer_subset_config(config,inputs=None,outputs=None):
    new_config = config.copy()
    model_config = new_config['config']
    layers = model_config['layers']
    
    if inputs is None:
        inputs = [inp[0] for inp in model_config['input_layers']]
    else:
        inputs = to_list(inputs)
    if outputs is None:
        outputs = [out[0] for out in model_config['output_layers']]
    else:
        outputs = to_list(outputs)

    layer_dict = {}
    for idx,layer in enumerate(layers):
        name = layer['name']
        layer_dict[name] = {}
        layer_dict[name]['config'] = layer
        layer_dict[name]['idx'] = idx
        
    idx_set = set()
    layer_name_list = outputs.copy()
    while len(layer_name_list) >0:
        layer_name = layer_name_list.pop(0)

        layer = layer_dict[layer_name]
        idx_set.add(layer['idx'])
        
        if layer_name in inputs:
            continue
        
        inbound_nodes = inbound_node_names(layer['config'])
        for node in inbound_nodes:
            node_idx = layer_dict[node]['idx']
            if node_idx not in idx_set:
                layer_name_list.append(node)
                
    idx_list = sorted(list(idx_set))
    new_layers = []
    for idx in idx_list:
        new_layers.append(layers[idx])
    new_config['config']['layers'] = new_layers
    new_config['config']['input_layers'] = [[inp,0,0] for inp in inputs]
    new_config['config']['output_layers'] = [[out,0,0] for out in outputs]
    return new_config
    
def inbound_node_names(layer_config):
    inbound_nodes = layer_config['inbound_nodes'][0]
    names = []
    for node in inbound_nodes:
        names.append(node[0])
    return names

def convert_layers_into_input_configs(config,inputs=None,batch_input_shapes=None):
    if inputs is None:
        return config
    inputs = to_list(inputs)
    batch_input_shapes = to_list(batch_input_shapes)
    
    new_config = config.copy()
    model_config = new_config['config']
    layers = model_config['layers']
    
    layer_dict = {}
    for idx,layer in enumerate(layers):
        name = layer['name']
        layer_dict[name] = {}
        layer_dict[name]['config'] = layer
        layer_dict[name]['idx'] = idx
        
    for inp,shape in zip(inputs,batch_input_shapes):
        layer = layer_dict[inp]
        idx = layer_dict[inp]['idx']
        new_input_template = {'class_name':'InputLayer',
                              'inbound_nodes':[],
                              'name':inp,
                              'config':{'dtype':'float32',
                                        'sparse':False,
                                        'name':inp,
                                        'batch_input_shape':shape
                                      }
                                }
        layers[idx] = new_input_template
    return new_config

def _deserialize_model(f, batch_size=1,custom_objects=None,
                       new_inputs=None,new_outputs=None,batch_input_shapes=None,
                       convert_to_queued=True,):

    model_config = f['model_config']
    if model_config is None:
        raise ValueError('No model found in config.')
    model_config = json.loads(model_config.decode('utf-8'))
    if new_inputs is not None or new_outputs is not None:
        model_config = layer_subset_config(model_config,new_inputs,new_outputs)
    model_config = convert_layers_into_input_configs(model_config,inputs=new_inputs,
                                                     batch_input_shapes=batch_input_shapes)
    if convert_to_queued:
        model_config = convert_to_queued_config(model_config,batch_size=batch_size)

    model = model_from_config(model_config, custom_objects=custom_objects)
    model_weights_group = f['model_weights']

    if 'keras_version' in model_weights_group:
        original_keras_version = model_weights_group['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in model_weights_group:
        original_backend = model_weights_group['backend'].decode('utf8')
    else:
        original_backend = None

    layer_names = model_weights_group['layer_names']

    layers = model.layers

    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)
            
    layer_dict = {}
    for layer in filtered_layers:
        if new_inputs is None or layer.name not in new_inputs:
            layer_dict[layer.name] = layer

    filtered_layer_names = []
    for name in layer_names:
        layer_weights = model_weights_group[name]
        weight_names = layer_weights['weight_names']
        if weight_names:
            filtered_layer_names.append(name)

    layer_names = filtered_layer_names


    weight_value_tuples = []
    for name, layer in layer_dict.items():
        layer_weights = model_weights_group[name]
        weight_names = layer_weights['weight_names']
        weight_values = [layer_weights[weight_name] for weight_name in weight_names]
        symbolic_weights = layer.weights
        weight_values = preprocess_weights_for_loading(layer,
                                                       weight_values,
                                                       original_keras_version,
                                                       original_backend,
                                                       reshape=False)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer '+
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)
    
    return model
