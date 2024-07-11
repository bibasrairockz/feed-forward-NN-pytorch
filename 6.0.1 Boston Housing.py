import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size= 19
input_size= 13
hidden_size= 500
output_size= 1
n_iters= 100
learing_rate= 0.001
p_step= 7
file_path= './data./boston/housing.csv'
data = []
with open(file_path, 'r') as f:
    for line in f:
        values = line.strip().split()
        data.append(values)
features = []
targets = []
for row in data:
    target = float(row[-1]) 
    feature = list(map(float, row[:-1]))  
    features.append(feature)
    targets.append(target)
features_tensor = torch.tensor(features, dtype=torch.float32)
features_numpy= features_tensor.numpy()
sc= StandardScaler()
features_numpy= sc.fit_transform(features_numpy)
features_tensor= torch.tensor(features_numpy, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)
targets_tensor= targets_tensor.view(targets_tensor.shape[0], 1)
X_train, X_test, Y_train, Y_test= train_test_split(features_tensor, targets_tensor, test_size= .15, random_state=1)
class BostonHousingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
train_dataset = BostonHousingDataset(X_train, Y_train)
test_dataset = BostonHousingDataset(X_test, Y_test)
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle=False)
example= iter(train_loader)
example_data, _= next(example)
class ffnn(nn.Module):
    def __init__(self, input_size, hidde_size, output_size):
        super(ffnn, self).__init__()
        self.l1= nn.Linear(input_size, hidde_size)
        self.relu= nn.ReLU()
        self.l2= nn.Linear(hidde_size, output_size)

    def forward(self, x):
        out= self.l1(x)
        out= self.relu(out)
        out= self.l2(out)
        return out
model= ffnn(input_size, hidden_size, output_size).to(device)
loss= nn.MSELoss()
optimizer= torch.optim.Adam(model.parameters(), lr= learing_rate)

def show_data():
    df_torch = pd.DataFrame(features_tensor.numpy(), columns=[f'feature_{i}' for i in range(1, 14)])
    print(df_torch.head())
    sns.pairplot(df_torch)
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

            if (i+1)% p_step == 0:
                print(f"Epoch [{epoch}/{n_iters}] | Steps [{i+1}/{n_steps}] : COST= {cost.item():.4f}")

def accuracy():
    with torch.no_grad():
        r2=0
        for (data, labels) in test_loader:
            data= data.reshape(-1, input_size).to(device)
            labels= labels.to(device)
            Y_pred= model(data)
            r= r2_score(labels.cpu().detach().numpy(), Y_pred.cpu().detach().numpy())
            r2+= r
        acc= (r2/len(test_loader))
        print(f'R2 score= {acc:.2f}')

if __name__=="__main__":
    show_data()
    train()
    accuracy()
    pass
     

