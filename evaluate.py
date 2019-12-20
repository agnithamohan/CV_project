import torch
import os
import pickle
import sys

#from data import GanDataset
#from torch.utils.data import DataLoader
from probabilistic_unet import ProbabilisticUnet
from model_helpers import getModelFilePath, getLatestModelEpoch, loadModel, saveModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-L", "--layer", help="the fpn layer to train")
parser.add_argument("-M", "--model", help="the model checkpoint to load")
parser.add_argument("-I", "--input", help="the input file to evaluate")
parser.add_argument("-O", "--output", help="the output file")
args = parser.parse_args()

if not args.layer:
	print("Provide the layer name in --layer")
	sys.exit()
if not args.model:
	print("Provide the model checkpoint file using --model")
	sys.exit()
if not args.input:
	print("Provide input file to evaluate")
	sys.exit()


latent_dims_layer = {
    'fpn_res5_2_sum': 10,
    'fpn_res4_5_sum': 20,
    'fpn_res3_3_sum': 10,
    'fpn_res2_2_sum': 100
}

fcomb_layer = {
    'fpn_res5_2_sum': 8,
    'fpn_res4_5_sum': 4,
    'fpn_res3_3_sum': 8,
    'fpn_res2_2_sum': 4
}
LAYER = args.layer

net = ProbabilisticUnet(input_channels=256, num_classes=256, num_filters=[256, 512, 1024, 2048], latent_dim=latent_dims_layer[LAYER], no_convs_fcomb=fcomb_layer[LAYER], beta=10.0, layer=LAYER)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
loadModel(net, optimizer, args.model)
print("Reading input from:", args.input)
inp = torch.load(args.input, map_location="cpu")
net.forward(inp, training=False)
out = net.sample(testing=True)
if(args.output):
	print("Output saved to:", args.output)
	torch.save(out, args.output)

print(inp.shape, out.shape)
