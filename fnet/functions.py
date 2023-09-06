import importlib
import json
import os
import pdb
import sys
import fnet
import torch

def load_model(path_model, gpu_ids=0, module='fnet_model'):
    module_fnet_model = importlib.import_module('fnet.' + module)
    if os.path.isdir(path_model):
        path_model = os.path.join(path_model, 'model.p')
    model = module_fnet_model.Model()
    model.load_state(path_model, gpu_ids=gpu_ids)
    return model

def load_model_from_dir(path_model_dir, in_dim, out_dim, gpu_ids=0, id=1):
    assert os.path.isdir(path_model_dir)
    path_model_state = os.path.join(path_model_dir, 'model.p')
    if id == 0:
      model = fnet.fnet_model.Model()
    else:
      model = fnet.fnet_model_2.Model()
    model.init_weights = False
    model.load_state(path_model_state, in_dim, out_dim, gpu_ids=gpu_ids)
    return model

def load_model_from_dir_li(path_model_dir, in_dim, out_dim, gpu_ids=0, id=1):
    assert os.path.isdir(path_model_dir)
    path_model_state = os.path.join(path_model_dir, 'model.p')
    model = fnet.fnet_model_si.Model()
    model.init_weights = False
    model.load_state(path_model_state, in_dim, out_dim, gpu_ids=gpu_ids)
    return model

def load_model_from_checkpoint(path_model_state, in_dim, out_dim, gpu_ids=0, id=1):
    # assert os.path.isdir(path_model_state)
    if id == 0:
      model = fnet.fnet_model.Model()
    else:
      model = fnet.fnet_model_2.Model()
    model.init_weights = False
    model.load_state(path_model_state, in_dim, out_dim, gpu_ids=gpu_ids)
    return model

def load_pretrained_model(path_model_state, in_dim, out_dim, out_drop, gpu_ids=0, id=1):
    # model = fnet.fnet_model_4.Model()
    # model.init_weights = False
    # state_dict = model.load_state(path_model_state, in_dim, out_dim, gpu_ids=gpu_ids)
    state_dict = torch.load(path_model_state)

    device = 'cuda'

    weight_input_layer = state_dict['nn_net_state']['input_layer.0.weight']
    w = torch.empty(weight_input_layer[:,0,:,:].size())
    w = torch.nn.init.xavier_uniform_(w)
    w = w[:, None, :, :].to(device)
    weight_input_layer = torch.cat((weight_input_layer, w), 1)

    weight_skip_layer = state_dict['nn_net_state']['input_skip.0.weight']
    w2 = torch.empty(weight_skip_layer[:,0,:,:].size())
    w2 = torch.nn.init.xavier_uniform_(w2)
    w2 = w2[:, None, :, :].to(device)
    weight_skip_layer = torch.cat((weight_skip_layer, w2), 1)

    bias_input_layer = state_dict['nn_net_state']['input_layer.0.bias']
    if len(bias_input_layer.size()) == 1:
        bias_input_layer = bias_input_layer[None,:]
    b = torch.empty((1, bias_input_layer.size()[-1]))
    b = torch.nn.init.zeros_(b).to(device)
    bias_input_layer = torch.cat((bias_input_layer, b),0)

    weight_output_layer = state_dict['nn_net_state']['output_layer.0.weight']
    weight_output_layer = torch.cat([weight_output_layer[:out_drop], weight_output_layer[out_drop+1:]])
    bias_output_layer = state_dict['nn_net_state']['output_layer.0.bias']
    bias_output_layer = torch.cat([bias_output_layer[:out_drop], bias_output_layer[out_drop+1:]])

    state_dict['nn_net_state']['input_layer.0.weight'] = weight_input_layer
    # state_dict['nn_net_state']['input_layer.0.bias'] = bias_input_layer
    state_dict['nn_net_state']['input_skip.0.weight'] = weight_skip_layer
    state_dict['nn_net_state']['output_layer.0.weight'] = weight_output_layer
    state_dict['nn_net_state']['output_layer.0.bias'] = bias_output_layer

    return state_dict
    