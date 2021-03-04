import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import codecs
from torch import nn
from torch import optim
import torch.nn.functional as F

def load_wsd_train_x():
    wsd_train_x = codecs.open('27236_train_data', mode = 'r', encoding= 'utf-8')
    line = wsd_train_x.readline()
    list1 = []
    while line:
        a = line.split()
        b = a[3:]
        list1.append(b)
        line = wsd_train_x.readline()
    return np.array(list1)
    wsd_train_x.close()


def load_wsd_test_x():
    wsd_test_x = codecs.open('27236_test_data', mode = 'r', encoding= 'utf-8')
    line = wsd_test_x.readline()
    list1 = []
    while line:
        a = line.split()
        b = a[3:]
        list1.append(b)
        line = wsd_test_x.readline()
    return np.array(list1)
    wsd_test_x.close()


def load_wsd_train_y():
    wsd_train_y = codecs.open('27236_train_target', mode = 'r', encoding = 'utf-8')
    line = wsd_train_y.readline()
    list1 = []
    while line:
        a = line.split()
        b = a[1:2]
        list1.append(b)
        line = wsd_train_y.readline()
    return (np.array(list1)).reshape(50,)
    wsd_train_y.close()



def load_wsd_test_y():
    wsd_test_y = codecs.open('27236_test_target', mode = 'r', encoding = 'utf-8')
    line = wsd_test_y.readline()
    list1 = []
    while line:
        a = line.split()
        b = a[1:2]
        list1.append(b)
        line = wsd_test_y.readline()
    return (np.array(list1)).reshape(50,)
    wsd_test_y.close()

wsd_train_x = load_wsd_train_x().astype(float)
wsd_test_x = load_wsd_test_x().astype(float)

wsd_train_y = load_wsd_train_y().astype(float)
wsd_test_y = load_wsd_test_y().astype(float)

max_epoch = 100
train_size = wsd_train_x.shape[0]
batch_size = 10
n_batch = train_size // batch_size

gogi_num = 6

class DealTrainDataSet(Dataset):
    def __init__(self):
        self.train_data = torch.from_numpy(wsd_train_x)
        self.train_target = torch.from_numpy(wsd_train_y)
        self.len = wsd_train_x.shape[0]
    def __getitem__(self, index):
        return self.train_data[index], self.train_target[index]

    def __len__(self):
        return self.len

class DealTestDataSet(Dataset):
    def __init__(self):
        self.test_data = torch.from_numpy(wsd_test_x)
        self.test_target = torch.from_numpy(wsd_test_y)
        self.len = wsd_test_x.shape[0]

    def __getitem__(self, index):
        return self.test_data[index], self.test_target[index]

    def __len__(self):
        return self.len

dealTrainDataSet = DealTrainDataSet()
train_loader = DataLoader(dataset = dealTrainDataSet, batch_size = batch_size, shuffle = True)
dealTestDataSet = DealTestDataSet()
test_loader = DataLoader(dataset = dealTestDataSet, batch_size = batch_size, shuffle = False)

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        x = self.layer(x)
        return x

def train():
    model = MyModel(768, gogi_num)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    cost = nn.CrossEntropyLoss()
    # train
    for epoch in range(max_epoch):
        max_acc = 0
        correct = 0
        running_loss = 0.0
        for train_data, train_target in train_loader:
            train_data, train_target = train_data.float(), train_target.long()
            train_data, train_target = Variable(train_data), Variable(train_target)
            optimizer.zero_grad()
            outputs = model(Variable(train_data))
            loss = cost(outputs, train_target)
            loss.backward()
            optimizer.step()
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(train_target.data.view_as(pred)).cpu().sum()
            running_loss /= len(train_loader.dataset)
            accuracy = float(correct) / train_size
        if max_acc < accuracy:
            max_acc = accuracy
            torch.save(model, "27236_wsd_max_model.pkl")
            torch.save(model.state_dict(), '27236_wsd_max_model_params.pkl')
        


def reload_model():
    train_model = torch.load('27236_wsd_max_model.pkl')
    return train_model

def test():
    test_loss = 0
    correct = 0
    model = reload_model()
    for test_data, test_target in test_loader:
        test_data, test_target = test_data.float(), test_target.long()
        test_data, test_target = Variable(test_data), Variable(test_target)
        outputs = model(Variable(test_data))
        # sum up batch loss

        test_loss += F.nll_loss(outputs, test_target, reduction='sum').item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(test_target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
    print('27236 Best Test Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    train()
    reload_model()
    test()