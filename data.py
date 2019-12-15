import os
from torch.utils.data import DataLoader, Dataset
import torch

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
        if len(self.inputFiles) != len(self.gtFiles):
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


def getSmallerDataLoader(dataloader, numSamples):
    smallerDataloader = []
    it = iter(train_loader)
    for i in numSamples:
        smallerDataloader.append(next(it))
    return smallerDataloader
