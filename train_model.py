import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.sampler import SubsetRandomSampler
# from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import pickle
from torchsummary import summary

class GanDataset(Dataset):
    def __init__(self, inputsDir, gtDir, layerName):
        # Possible layers: fpn_res5_2_sum, fpn_res4_5_sum, fpn_res3_3_sum, fpn_res2_2_sum
        self.inputsDir = inputsDir
        self.gtDir = gtDir
        self.layerName = layerName
        self.numFiles = 0
        self.files = []
        self.inputFiles = []
        self.gtFiles = []
        self.scanDir()

    def scanDir(self):
        self.inputFiles = [f for f in os.listdir(os.path.join(self.inputsDir, self.layerName)) if f.endswith(".pt")]
        self.gtFiles = [f for f in os.listdir(os.path.join(self.gtDir, self.layerName)) if f.endswith(".pt")]
        print("Found", len(self.inputFiles), "files for 'input'")
        print("Found", len(self.gtFiles), "files for 'gt'")
        if(len(self.inputFiles) is not len(self.gtFiles)):
            print("Inconsistency in count of inputs and gt")
        self.numFiles = len(self.inputFiles)


    def __len__(self):
        return self.numFiles

    def __getitem__(self, idx):
        if(idx >= len(self.inputFiles)):
            raise Exception("Index requested is greater than files available")
        filename = self.inputFiles[idx]
        inputFilepath = os.path.join(self.inputsDir, self.layerName, filename)
        gtFilepath = os.path.join(self.gtDir, self.layerName, filename)

        inpt = torch.load(inputFilepath, map_location="cpu")
        gt = torch.load(gtFilepath, map_location="cpu")

        sample = {'input': inpt, 'gt': gt}
        return sample

dataset = GanDataset("/scratch/amr1215/gan_dataset/train/inputs", "/scratch/amr1215/gan_dataset/train/gt", "fpn_res5_2_sum")
# print(dataset[0])
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
# for idx, data in enumerate(train_loader):
#     inp = data["input"][0][0]
#     gt = data["gt"][0][0]
#     # print("INPUT", inp.shape)
#     # print("GT",gt.shape)
#     # inp.gpu()
#     break




# dataset = LIDC_IDRI(dataset_location = 'data/')
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(0.1 * dataset_size))
# np.random.shuffle(indices)
# train_indices, test_indices = indices[split:], indices[:split]
# train_sampler = SubsetRandomSampler(train_indices)
# test_sampler = SubsetRandomSampler(test_indices)
# train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
# test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
# print("Number of training/test patches:", (len(train_indices),len(test_indices)))

net = ProbabilisticUnet(input_channels=256, num_classes=256, num_filters=[256, 512, 1024, 2048], latent_dim=10, no_convs_fcomb=8, beta=10.0)
net.to("cuda")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 1000
#smallerTrainloader = []
#for idx, data in enumerate(train_loader):
#    smallerTrainloader.append(data)
#    if idx==0:
#        break

# print(len(smallerTrainloader))
model_dest = '/scratch/amr1215/probunet_models/'
for epoch in range(epochs):
    # for step, (patch, mask, _) in enumerate(train_loader): 
    for idx, data in enumerate(train_loader):
        print("epoch:", epoch, "idx:", idx)
        inp = data["input"][0].cuda()
        gt = data["gt"][0].cuda()
        print("HAHAHAHA LOSS:", torch.nn.L1Loss()(inp, gt).item())

        net.forward(inp, gt, training=True)
        # print(summary(net, (1, 256, 32, 64)))

        elbo = net.elbo(gt)
        l2posterior = l2_regularisation(net.posterior)
        l2prior = l2_regularisation(net.prior)
        l2fcomb = l2_regularisation(net.fcomb.layers)
        # print("ELBO Loss:" , elbo.item())
        # print("L2 Posterior:", l2posterior)
        # print("L2 Prior:", l2prior)
        # print("L2 FComb:", l2fcomb)
        reg_loss = l2posterior + l2prior + l2fcomb
        loss = -elbo + 1e-5 * reg_loss
        print("Total Loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    if(epoch%5 == 0):
	model_path = model_dest+'model_' + str(epoch) + '.pth'
	torch.save(net,model_path)
	    
	#print("MODEL WEIGHTS") 			
    #for param_tensor in net.state_dict():
    #	print(param_tensor, "\t", net.state_dict()[param_tensor].size())
	
#         patch = patch.to(device)
#         mask = mask.to(device)
#         mask = torch.unsqueeze(mask,1)
        
#         
#         reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
#         loss = -elbo + 1e-5 * reg_loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
