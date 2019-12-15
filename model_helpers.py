import os
import torch

def getModelFilePath(basePath, layer, epoch):
    return os.path.join(basePath, layer, 'model_' + str(epoch) + '.pth')

def getLatestModelEpoch(basePath, layer):
    modelFiles = [f for f in os.listdir(os.path.join(basePath, layer)) if (f.startswith('model_') and f.endswith(".pth"))]
    if modelFiles:
        epoch = [int(f.split('.')[-2].split('_')[1]) for f in modelFiles]
        return max(epoch)
    else:
        return None

def loadModel(model, optimizer, loadPath):
    checkpoint = torch.load(loadPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def saveModel(model, optimizer, savePath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, savePath)