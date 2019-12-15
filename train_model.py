import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from probabilistic_unet import ProbabilisticUnet
from data import GanDataset
from utils import l2_regularisation

MODELS_DEST = '/scratch/amr1215/probunet_models/'
LAYER = "fpn_res5_2_sum"
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

# CREATE DATALOADERS
train_dataset = GanDataset(GAN_DATASET['TRAIN']['INPUT'], GAN_DATASET['TRAIN']['GT'], LAYER)
eval_dataset = GanDataset(GAN_DATASET['EVAL']['INPUT'], GAN_DATASET['EVAL']['GT'], LAYER)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=4)

# INITIALISE NETWORKS
net = ProbabilisticUnet(input_channels=256, num_classes=256, num_filters=[256, 512, 1024, 2048], latent_dim=10, no_convs_fcomb=8, beta=10.0).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

# LOAD PRESAVED MODEL IF EXISTS
currEpoch = 0
max_epochs = 1000
savedEpoch = getLatestModelEpoch(MODELS_DEST, LAYER)
if savedEpoch:
    loadModel(net, optimizer, getModelFilePath(MODELS_DEST, LAYER, savedEpoch))
    currEpoch = savedEpoch+1
    

for epoch in range(currEpoch, max_epochs):
    # TRAINING
    net.train()
    for idx, data in enumerate(train_loader):
        print("Epoch:", epoch, "idx:", idx)
        inp = data["input"][0].cuda()
        gt = data["gt"][0].cuda()
        print("Target Loss:", torch.nn.L1Loss()(inp, gt).item())
        net.forward(inp, gt, training=True)
        elbo = net.elbo(gt)
        l2posterior = l2_regularisation(net.posterior)
        l2prior = l2_regularisation(net.prior)
        l2fcomb = l2_regularisation(net.fcomb.layers)
        reg_loss = l2posterior + l2prior + l2fcomb
        loss = -elbo + 1e-5 * reg_loss
        print("Total Loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # SAVE CHECKPOINT
    if epoch % 2 == 0:
        saveModel(net, optimizer, getModelFilePath(MODELS_DEST, LAYER, epoch))
    
    # EVALUATION 
    net.eval()
    totalLoss = 0.0
    count = 0
    betterCount = 0
    for idx, data in enumerate(eval_loader):
        print("EVAL idx:", idx)
        inp = data["input"][0].cuda()
        gt = data["gt"][0].cuda()
        lossBaseline = torch.nn.L1Loss()(inp, gt).item()
        print("Target Loss:", lossBaseline)
        net.forward(inp, gt, training=True)
        elbo = net.elbo(gt)
        l2posterior = l2_regularisation(net.posterior)
        l2prior = l2_regularisation(net.prior)
        l2fcomb = l2_regularisation(net.fcomb.layers)
        reg_loss = l2posterior + l2prior + l2fcomb
        loss = -elbo + 1e-5 * reg_loss
        print("Total Loss: ", loss.item())
        if(loss.item() < lossBaseline):
            betterCount += 1
        count += 1
        totalLoss += loss.item()
    print("EVAL Total Loss: ", totalLoss)
    print("EVAL Average Loss: ", totalLoss/count)
    print("EVAL Better than Target Loss: ", betterCount, "/", count)
        