import os
import argparse
import torch
import torch.nn as nn
#CLASSI
class InvertedLinearRegressionModelV2(nn.Module):
    def __init__(self,activation_func_reg,n_nodes,n_layers,dim_input,dim_output,device,division):
        super(InvertedLinearRegressionModelV2, self).__init__()
        
        modules=[]
        if activation_func_reg=="ELU":
            modules.append(torch.nn.Linear(in_features=dim_input,out_features=n_nodes))
            modules.append(torch.nn.ELU())
        if activation_func_reg=="Tanh":
            modules.append(torch.nn.Linear(in_features=dim_input,t_features=n_nodes))
            modules.append(torch.nn.Tanh())
        if activation_func_reg=="ReLU":
            modules.append(torch.nn.Linear(in_features=dim_input,out_features=n_nodes))
            modules.append(torch.nn.ReLU())
        if activation_func_reg=="Leaky_ReLU":
            modules.append(torch.nn.Linear(in_features=dim_input,out_features=n_nodes))
            modules.append(torch.nn.LeakyReLU())
        if activation_func_reg=="Sigmoid":
            modules.append(torch.nn.Linear(in_features=dim_input,out_features=n_nodes))
            modules.append(torch.nn.Sigmoid())
        
        hidden_units=n_nodes
        for i in range(n_layers-1):
            if activation_func_reg=="ELU":
                modules.append(torch.nn.Linear(hidden_units,int(hidden_units*division)))
                modules.append(torch.nn.ELU())
                hidden_units = int(hidden_units*division)
                print(f"hidden units = {hidden_units}")
            if activation_func_reg=="Tanh":
                modules.append(torch.nn.Linear(hidden_units,int(hidden_units*division)))
                modules.append(torch.nn.Tanh())
                hidden_units = int(hidden_units*division)
            if activation_func_reg=="ReLU":
                modules.append(torch.nn.Linear(hidden_units,int(hidden_units*division)))
                modules.append(torch.nn.ReLU())
                #hidden_units = int(hidden_units*division)
                print(f"hidden units = {hidden_units}")
                hidden_units = int(hidden_units*division)
            if activation_func_reg=="Leaky_ReLU":
                modules.append(torch.nn.Linear(hidden_units,int(hidden_units*division)))
                modules.append(torch.nn.LeakyReLU())
                hidden_units = int(hidden_units*division)
            if activation_func_reg=="Sigmoid":
                modules.append(torch.nn.Linear(hidden_units, int(hidden_units*division)))
                modules.append(torch.nn.Sigmoid())
                hidden_units = int(hidden_units*division)
        modules.append(torch.nn.Linear(hidden_units,dim_output))

        self.linear_layer_stack=nn.Sequential(*modules)
        self.device=device
    def forward(self,x: torch.Tensor) ->torch.Tensor:
        #result=self.linear_layer_stack(x)
        #result=torch.where(result>0.4,torch.tensor(0.8),torch.tensor(0.0))
        #return result
        return self.linear_layer_stack(x)

def crea_cartella(nome_cartella):
    try:
        os.makedirs(nome_cartella)
        print(f"Cartella '{nome_cartella}' creata con successo.")
    except FileExistsError:
        print(f"La cartella '{nome_cartella}' esiste già.")

def logic_prog(epochs, nodes_dim, learning_rate, batch_size,activation_function,n_layers):
    # La logica del tuo programma va qui
    print(f"Epochs: {epochs}")
    print(f"Nodes Dimension: {nodes_dim}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Activation Function: {activation_function}")
    print(f"n_layers: {n_layers}")
    return epochs, nodes_dim, learning_rate, batch_size,activation_function,n_layers

