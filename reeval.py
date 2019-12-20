import torch
import os
import pickle

from data import GanDataset
from torch.utils.data import DataLoader
from probabilistic_unet import ProbabilisticUnet
from model_helpers import getModelFilePath, getLatestModelEpoch, loadModel, saveModel

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
LAYER="fpn_res5_2_sum"

eval_dataset = GanDataset(GAN_DATASET['EVAL']['INPUT'], GAN_DATASET['EVAL']['GT'], LAYER)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=0)

net = ProbabilisticUnet(input_channels=256, num_classes=256, num_filters=[256, 512, 1024, 2048], latent_dim=10, no_convs_fcomb=8, beta=10.0).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

currEpoch = 0
max_epochs = 110

net.eval()
for epoch in range(currEpoch, max_epochs, 2):
    loadModel(net, optimizer, getModelFilePath(MODELS_DEST, LAYER, epoch))
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
    betterCount = 0
    for idx, data in enumerate(eval_loader):
    #if idx == 2:
    #    break
        #print("EVAL idx:", idx)
        inp = data["input"][0].cuda()
        gt = data["gt"][0].cuda()
        lossBaseline = torch.nn.L1Loss()(inp, gt)
        print("Target Loss:", lossBaseline.item())
        if(torch.isnan(lossBaseline)):
            continue
        net.forward(inp, gt, training=False)
        reconLoss, klLoss = net.elbo(gt,training=False)
        loss = reconLoss
        print("Total Loss: ", loss.item())
        if(loss.item() < lossBaseline.item()):
            betterCount += 1
        eval_losses['rec'].append(reconLoss.item())
        eval_losses['kl'].append(0.0)
        eval_losses['l2pos'].append(0.0)
        eval_losses['l2pri'].append(0.0)
        eval_losses['l2fcom'].append(0.0)
        eval_losses['total'].append(loss.item())
        eval_targetLosses.append(lossBaseline.item())
        eval_count += 1
        eval_totalLoss += loss.item()
    print("Epoch:", epoch)
    print("EVAL Total Loss: ", eval_totalLoss)
    print("EVAL Average Loss: ", eval_totalLoss/eval_count)
    print("EVAL Better than Target Loss: ", betterCount, "/", eval_count)
    with open(getModelFilePath(MODELS_DEST, LAYER, epoch).split('.')[0]+'_eval_loss', 'wb') as f:
        pickle.dump({'losses': eval_losses, 'targetLosses': eval_targetLosses, 'count': eval_count, 'betterCount': betterCount}, f)


    
