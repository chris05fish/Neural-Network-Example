import torch
import torchvision 
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

train = datasets.MNIST("", train=True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

#for data in trainset:
    #print(data)
    #break

# data is last element in above forloop
#plt.imshow(data[0][0].view(28,28))
# .view reshapes to fit 28 by 28, instead of origional 1 by 28 by 28
# show an example number in the dataset
#plt.show()

# make sure data is balenced
#total = 0
#counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

#for data in trainset:
    #Xs, ys = data
    #for y in ys:
        #counter_dict[int(y)] += 1

#print(counter_dict)
# shows we have about 6,000 of each digit 0-9

class Net(nn.Module):
    def __init__(self):
        # inheritence from other class so run init
        super().__init__()
        # 784 = 28 x 28 images (input), 64 = output
        # nn.Linear == fully connected 
        # input layer
        self.fc1 = nn.Linear(784, 64)
        # hidden layers  
        # input layer to 1st hidden layer has to match nodes 64->64
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        # output layer 10 nodes (0-9)
        self.fc4 = nn.Linear(64, 10) 

    # how to run data through nn
    def forward(self, x):
        # rectivative linear (relu) run activation function through layer
        x = F.relu(self.fc1(x))   
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # for outputlayer to distribute across tensor dim 1
        return F.log_softmax(x, dim=1)

net = Net()

# pass data through network
#X = torch.rand((28,28))
# format in a way the library wants it (-1 says input will be of unknown shape)
#X = X.view(-1,28*28)
# pass data through nn
#output = net(X)

optimizer = optim.Adam(net.parameters(), lr=0.001)

# amount of times passed data through
EPOCHS = 3

# pass data and train model
for epoch in range(EPOCHS):
    for data in trainset:
        # data is a bunch of featuresets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

# calculate accuracy
correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1,784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

# show pic of a number in the dataset
plt.imshow(X[0].view(28,28))
plt.show()

print(torch.argmax(net(X[0].view(-1,784))[0]))