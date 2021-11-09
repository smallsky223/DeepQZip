# -*- coding: utf-8 -*-
import torch as t
import torch
import argparse
from models_cpp import QVRNNCpp

parser = argparse.ArgumentParser()
parser.add_argument('-data_params', action='store', 
                    dest='params_file',
                    help='params file')
parser.add_argument('-model', action='store',
                    dest='model_file_temp',
                    help='weights will be stored with this name')
parser.add_argument('-model_pt', action='store',
                    dest='model_file',
                    help='weights pt will be stored with this name')
arguments = parser.parse_args()

with open(arguments.params_file, 'r') as f:
        alphabet_size = int(f.readline())

if __name__ == '__main__':
    model = QVRNNCpp(alphabet_size)
    print(model)
    model.load_state_dict(t.load(arguments.model_file_temp, map_location='cuda:0'))
    model = model.cuda()
    model.eval()
    example_x = torch.randint(0,alphabet_size,(128,64)).type(torch.LongTensor).cuda()
    example_q = torch.randint(0,94,(128,64)).type(torch.LongTensor).cuda()
    traced_script_module = torch.jit.trace(model,(example_x, example_q))
    traced_script_module.save(arguments.model_file)
