import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size= 100
input_size= 784
hidden_size= 500
output_size= 10
learning_rate= 0.001
n_iters= 2
train_dataset= torchvision.datasets.MNIST(root='./data',
                                          train= True,
                                          transform= transforms.ToTensor(),
                                          download= True
                                          )
test_dataset= torchvision.datasets.MNIST(root='./data',
                                         train= False,
                                         transform= transforms.ToTensor()
                                         )
train_loader= torch.utils.data.DataLoader(dataset= train_dataset,
                                          shuffle= True,
                                          batch_size= batch_size)
test_loader= torch.utils.data.DataLoader(dataset= test_dataset,
                                         shuffle= False,
                                         batch_size= batch_size)
class ffnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ffnn, self).__init__()
        self.linear1= nn.Linear(input_size, hidden_size)
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out= self.linear1(x)
        out= self.relu(out)
        out= self.linear2(out)
        return out
model = ffnn(input_size, hidden_size, output_size).to(device)
loss= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr= learning_rate)

def show_data():
    example= iter(train_loader)
    example_data, example_label= next(example)
    # plt.imshow(example_data[0][0], cmap= 'gray')

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(example_data[i][0], cmap= 'gray')
    plt.show()

def train():
    n_steps= len(train_loader)
    for epoch in range(n_iters):
        for i, (data, labels) in enumerate(train_loader):
            data= data.reshape(-1, input_size).to(device)
            labels= labels.to(device)

            Y_pred= model(data)
            cost= loss(Y_pred, labels)

            optimizer.zero_grad()
            cost.backward()
            
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_iters}] | Step [{i+1}/{n_steps}] : COST= {cost.item():.4f}')

def accuracy():
    with torch.no_grad():
        n_samples= n_pred= 0
        for data, labels in test_loader:
            data= data.reshape(-1, input_size).to(device)
            labels= labels.to(device)
            Y_pred= model.forward(data)

            n_samples+= labels.shape[0]
            _, pred= torch.max(Y_pred, 1)
            n_pred+= (pred==labels).sum().item()
        acc= (n_pred/n_samples) * 100
        print(f'Accuracy= {acc:.2f} %')


if __name__== '__main__':
    # show_data()
    train()
    accuracy()
    pass
