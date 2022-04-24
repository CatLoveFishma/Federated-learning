import torch
import copy
from torch import nn
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def Normalize(data):
    min_number=min(data)
    if(min_number<0):
        data+=abs(min_number)
        min_number=min(data)
    max_number=max(data)
    max_distance=max_number-min_number
    norm_data=(data-min_number)/(max_distance)
    norm_data=(norm_data-0.5)/0.5
    return norm_data

def FedAvg(w): 
    w_avg = copy.deepcopy(w[0]) 
    #w_avg.to(t.float32)
    for k in w_avg.keys(): 
        for i in range(1, len(w)):
            w_avg[k].to(torch.float32)
            w[i][k].to(torch.float32)
            w_avg[k] =(w_avg[k]+ w[i][k] ).to(torch.float32)
        w_avg[k] = torch.div(w_avg[k], len(w))
        
    return w_avg 

class DatasetSplit(Dataset): #Dataset是标准class
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def test_model(net_g, dataset,idxs,args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available() and args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss