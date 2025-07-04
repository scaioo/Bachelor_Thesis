import os
import re
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sca_IO_func as sca
#Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Configura il parser degli argomenti da riga di comando
parser = argparse.ArgumentParser(description="Descrizione del tuo programma")

# Aggiungi gli argomenti che il programma accetta
parser.add_argument("--epochs", type=int, default=3000, help="Numero di epoche (default: 2000)")
parser.add_argument("--nodes_dim", type=int, default=1024, help="Dimensione dei nodi (default: 1024)")
parser.add_argument("--learning_rate", type=float, default=0.005, help="Tasso di apprendimento (default: 0.005)")
parser.add_argument("--batch_size", type=int, default=32, help="Dimensione del batch (default: 32)")
parser.add_argument("--activation_function", type=str, default="ReLU", help="Funzione di attivazione (default: ReLU)")
parser.add_argument("--n_layers", type=int, default=4, help="Numero di layer (default: 4)")
parser.add_argument("--division", type=float, default=0.125, help="divisione nodes dim (default: 0.125)")
# Analizza gli argomenti da riga di comando
args = parser.parse_args()
# Set Hyperparameters
epochs,nodes_dim,learning_rate,batch_size,activation_function,n_layers,division = sca.logic_prog(args.epochs, args.nodes_dim, args.learning_rate, args.batch_size,args.activation_function,args.n_layers,args.division)

#Creo una cartella dove salvare i risultati
directory=f"/home/scaioli/Simulazioni_NN/risultati_2HMG/epochs_{epochs}_nodes_dim_{nodes_dim}_lr_{learning_rate}_batch_{batch_size}_activ_func_{activation_function}_n_layers_{n_layers}_division_{division}"
sca.crea_cartella(directory)

#Apro file su cui scrivere output
file_path=f"/home/scaioli/Simulazioni_NN/risultati_2HMG/epochs_{epochs}_nodes_dim_{nodes_dim}_lr_{learning_rate}_batch_{batch_size}_activ_func_{activation_function}_n_layers_{n_layers}_division_{division}/output.txt"
with open(file_path, 'a') as file:
  file.write(f"Epochs = {epochs} | Nodes Dim = {nodes_dim} | Learning Rate = {learning_rate} | Batch size = {batch_size} | Activation function = {activation_function} | Number of Layers = {n_layers} | Division = {division}  \n")
print(f"Epochs = {epochs} | Nodes Dim = {nodes_dim} | Learning Rate = {learning_rate} | Batch size = {batch_size}| Activation function = {activation_function} | Number of Layers = {n_layers} | Division = {division} ")

# Lists to obtain data and associated times
data_list = []
time_list = []

#carico dataset
dataset = torch.load('/home/scaioli/Datasets/2HMG_dataset.pth')
#X=dataset[0]
#y=dataset[1]
X = torch.stack([sample[0] for sample in dataset])  # Concatena i tensori lungo una nuova dimensione
y = torch.stack([sample[1] for sample in dataset])  # Concatena i tensori lungo una nuova dimensione
print(f"shape dataset: X: {X.shape}, y: {y.shape}")
print(f"X: {X}, y: {y}")
# Create train and test splits
train_split = int(1 * len(X)) # 80% of data used for training set
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:] 
#parameters
dim_input=2277082
dim_output=1

# Put data on the target device (device agnostic code for data)
X_train =  X_train.to(device)
y_train =  y_train.to(device)
X_test =  X_test.to(device)
y_test =  y_test.to(device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

#set manual seed
model_1 = sca.LinearRegressionModelV2(activation_func_reg=activation_function,n_nodes=nodes_dim,n_layers=n_layers,dim_input=dim_input,dim_output=dim_output,device=device,division=division)
model_1.to(device)
# Setup a Loss function
loss_fn = nn.L1Loss() #same as MAE
#Setup our optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),lr=learning_rate,)

#Let's try a training loop
vect_train_loss = []
vect_test_loss = []

for epoch in range(epochs):
  ###Training
  train_loss = 0
  # Add a loop to loop through the training batches
  for batch, (X_train,y_train) in enumerate(train_loader):
    model_1.train()
    # 1. Forward pass
    y_pred = model_1(X_train)
    # 2. Calculate loss (per batch)
    loss = loss_fn(y_pred.squeeze(dim=1),y_train)
    #print(y_pred.shape, y_train.shape,X_train.shape)
    train_loss += loss # accumulate train loss
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # 4. Loss backward
    loss.backward()
    # 5. Optimizer step
    optimizer.step()
    #Print out what's happening
  if batch % 100 == 0 :
    with open(file_path, 'a') as file:
      file.write(f'Looked at {batch*len(X_train)}/({len(train_loader.dataset)}/({nodes_dim}/{n_layers})) samples \n')
    print(f'Looked at {batch*len(X_train)}/{len(train_loader.dataset)}/{nodes_dim}/{n_layers} samples')
  # Divide total trein loss by length od train dataloader
  train_loss /= len(train_loader)
  if epoch % 100 == 0 :
    with open(file_path,'a') as file:
      file.write(f"train loss: {train_loss}, len train_loader: {len(train_loader)} \n") 
    print(f"train loss: {train_loss}, len train_loader: {len(train_loader)}")
  #train_loss /= len(train_loader)
  vect_train_loss.append(train_loss.item())
  
  ### Testing
  test_loss=0
  model_1.eval()
  with torch.inference_mode():
    for X_test, y_test in test_loader:
      # 1. Forward pass
      test_pred = model_1(X_test)
      # 2. Calculate loss (accumulatively)
      test_loss+= loss_fn(test_pred.squeeze(dim=1), y_test)
      # Calculate the test loss average per batch
    test_loss/=len(test_loader)
    vect_test_loss.append(test_loss.item())
  # Print out what's happening
  if epoch % 100 == 0 :
    with open(file_path, 'a') as file:
      file.write( f"Epoch: {epoch} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f} \n ")
    print( f"Epoch: {epoch} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f} \n ")

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)
ax1.plot(vect_test_loss)
ax1.set_ylabel("test loss")
ax1.set_xlabel("epochs")
ax2.plot(vect_train_loss)
ax2.set_ylabel("training loss")
ax1.set_xlabel("epochs")
plt.savefig(f"/home/scaioli/Simulazioni_NN/risultati_2HMG/epochs_{epochs}_nodes_dim_{nodes_dim}_lr_{learning_rate}_batch_{batch_size}_activ_func_{activation_function}_n_layers_{n_layers}_division_{division}/Loss.png")

#Save the model
model_path=f"/home/scaioli/Simulazioni_NN/risultati_2HMG/epochs_{epochs}_nodes_dim_{nodes_dim}_lr_{learning_rate}_batch_{batch_size}_activ_func_{activation_function}_n_layers_{n_layers}_division_{division}/model.pth"
with open(file_path, 'a') as file:
  file.write( f"state dict del modello: {model_1.state_dict()} \n ")
print( f"state dict del modello: {model_1.state_dict()} \n ") 
torch.save(model_1.state_dict(),model_path)
