import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from probabilistic_unet import ProbabilisticUnet
from data import GanDataset
from utils import l2_regularisation
from model_helpers import getModelFilePath, getLatestModelEpoch, loadModel, saveModel
import pickle
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-L", "--layer", help="the fpn layer to train")
args = parser.parse_args()

LAYER = None

if not args.layer:
    print("Please supply layername using --layer")
    sys.exit()
else:
    LAYER = args.layer

MODELS_DEST = '/scratch/amr1215/probunet_models/'
GAN_DATASET = {
    'TRAIN': {
        'INPUT': "/scratch/amr1215/gan_dataset/train/inputs",
        'GT': "/scratch/amr1215/gan_dataset/train/gt"
    },
    'EVAL': {
        'INPUT': "/scratch/amr1215/gan_dataset/eval/inputs",
        'GT': "/scratch/amr1215/gan_dataset/eval/gt"
    }
}

latent_dims_layer = {
    'fpn_res5_2_sum': 10,
    'fpn_res4_5_sum': 10,
    'fpn_res3_3_sum': 10,
    'fpn_res2_2_sum': 100
}

fcomb_layer = {
    'fpn_res5_2_sum': 8,
    'fpn_res4_5_sum': 10,
    'fpn_res3_3_sum': 10,
    'fpn_res2_2_sum': 4
}

if not os.path.exists(os.path.join(MODELS_DEST, LAYER)):
    os.makedirs(os.path.join(MODELS_DEST, LAYER))

# CREATE DATALOADERS
train_dataset = GanDataset(GAN_DATASET['TRAIN']['INPUT'], GAN_DATASET['TRAIN']['GT'], LAYER)
eval_dataset = GanDataset(GAN_DATASET['EVAL']['INPUT'], GAN_DATASET['EVAL']['GT'], LAYER)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=0)

# INITIALISE NETWORKS
net = ProbabilisticUnet(input_channels=256, num_classes=256, num_filters=[256, 512, 1024, 2048], latent_dim=latent_dims_layer[LAYER], no_convs_fcomb=fcomb_layer[LAYER], beta=10.0, layer=LAYER).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

currEpoch = 0
max_epochs = 200
savedEpoch = getLatestModelEpoch(MODELS_DEST, LAYER)
if savedEpoch:
    loadModel(net, optimizer, getModelFilePath(MODELS_DEST, LAYER, savedEpoch))
    currEpoch = savedEpoch+1
    
for epoch in range(currEpoch, max_epochs):
    # TRAINING
    net.train()
    train_losses = {
        'rec': [],
        'kl': [],
        'l2pos': [],
        'l2pri': [],
        'l2fcom': [],
        'total': []
    }
    train_targetLosses = []
    train_count = 0
    for idx, data in enumerate(train_loader):
        # print("Epoch:", epoch, "idx:", idx)
        inp = data["input"][0].cuda()
        gt = data["gt"][0].cuda()
        targetLoss = torch.nn.L1Loss()(inp, gt)
        print("Target Loss:", targetLoss.item())
        # Extremely important to protect from initial KL collapse
        if(torch.isnan(targetLoss)):
            continue
        net.forward(inp, gt, training=True)
        reconLoss, klLoss = net.elbo(gt)
        elbo = -(reconLoss + 10.0 * klLoss)
        l2posterior = l2_regularisation(net.posterior)
        l2prior = l2_regularisation(net.prior)
        l2fcomb = l2_regularisation(net.fcomb.layers)
        reg_loss = l2posterior + l2prior + l2fcomb
        loss = -elbo + 1e-5 * reg_loss
        if(loss.item() > 100000):
            continue
        print("Total Loss: ", loss.item())
        train_losses['rec'].append(reconLoss.item())
        train_losses['kl'].append(klLoss.item())
        train_losses['l2pos'].append(l2posterior.item())
        train_losses['l2pri'].append(l2prior.item())
        train_losses['l2fcom'].append(l2fcomb.item())
        train_losses['total'].append(loss.item())
        train_targetLosses.append(targetLoss.item())
        train_count += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with open(getModelFilePath(MODELS_DEST, LAYER, epoch).split('.')[0]+'_train_loss', 'wb') as f:
        pickle.dump({'losses': train_losses, 'targetLosses': train_targetLosses}, f)
    
    # SAVE CHECKPOINT
    if epoch % 2 == 0:
        saveModel(net, optimizer, getModelFilePath(MODELS_DEST, LAYER, epoch))
    
    # EVALUATION 
    net.eval()
    eval_totalLoss = 0.0
    eval_losses = {
        'rec': [],
        'kl': [],
        'l2pos': [],
        'l2pri': [],
        'l2fcom': [],
        'total': []
    }
    eval_targetLosses = []
    eval_count = 0
    for idx, data in enumerate(eval_loader):
	    # print("EVAL idx:", idx)
        inp = data["input"][0].cuda()
        gt = data["gt"][0].cuda()
        lossBaseline = torch.nn.L1Loss()(inp, gt).item()
        # print("Target Loss:", lossBaseline)
        net.forward(inp, gt, training=False)
        reconLoss, klLoss = net.elbo(gt,training=False)
        elbo = -(reconLoss + 10.0 * klLoss)
        l2posterior = l2_regularisation(net.posterior)
        l2prior = l2_regularisation(net.prior)
        l2fcomb = l2_regularisation(net.fcomb.layers)
        reg_loss = l2posterior + l2prior + l2fcomb
        loss = -elbo + 1e-5 * reg_loss
        # print("Total Loss: ", loss.item())
        eval_losses['rec'].append(reconLoss.item())
        eval_losses['kl'].append(klLoss.item())
        eval_losses['l2pos'].append(l2posterior.item())
        eval_losses['l2pri'].append(l2prior.item())
        eval_losses['l2fcom'].append(l2fcomb.item())
        eval_losses['total'].append(loss.item())
        eval_targetLosses.append(lossBaseline)
        eval_count += 1
        eval_totalLoss += loss.item()
    print("Epoch:", epoch)
    print("EVAL Total Loss: ", eval_totalLoss)
    print("EVAL Average Loss: ", eval_totalLoss/eval_count)
    with open(getModelFilePath(MODELS_DEST, LAYER, epoch).split('.')[0]+'_eval_loss', 'wb') as f:
        pickle.dump({'losses': eval_losses, 'targetLosses': eval_targetLosses}, f)
