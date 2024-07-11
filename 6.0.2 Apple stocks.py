import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

ticker = 'AAPL'
df = yf.download(ticker, start='2023-01-01', end='2024-01-01', interval='1d')
df = df.reset_index()
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df = df.drop(columns=['Date'])
X_np= df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'DayOfWeek',  'Month',  'Quarter' ]].values  
Y_np = df['Close'].values
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size= 25
input_size= 8
hidden_size1= 64
hidden_size2= 32
output_size= 1
n_iters= 50
learing_rate= 0.01
p_step= 3
sc= StandardScaler()
X_np= sc.fit_transform(X_np)
X_tn= torch.tensor(X_np, dtype=torch.float32)
Y_tn = torch.tensor(Y_np, dtype=torch.float32)
print(X_tn.shape, type(Y_tn))
Y_tn= Y_tn.view(Y_tn.shape[0], 1)
X_train, X_test, Y_train, Y_test= train_test_split(X_tn, Y_tn, test_size= .1, random_state=1)
class AppleData(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
train_dataset = AppleData(X_train, Y_train)
test_dataset = AppleData(X_test, Y_test)
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle=True)

class ffnn(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ffnn, self).__init__()
        self.l1= nn.Linear(input_size, hidden_size1)
        self.relu= nn.ReLU()
        self.l2= nn.Linear(hidden_size1, hidden_size2)
        self.l3= nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out= self.l1(x)
        out= self.relu(out)
        out= self.l2(out)
        out= self.relu(out)       
        out= self.l3(out)
        return out
model= ffnn(input_size, hidden_size1, hidden_size2, output_size).to(device)

loss= nn.MSELoss()
optimizer= torch.optim.Adam(model.parameters(), lr= learing_rate)

def show_data():
    df_torch = pd.DataFrame(X_tn.numpy(), columns=[f'feature_{i}' for i in range(1, 9)])
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

if __name__=='__main__':
    # show_data()
    train()
    accuracy()
    
    pass