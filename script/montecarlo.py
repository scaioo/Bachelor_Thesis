import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from scipy.stats import linregress
import torch
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import sys
sys.path.append('/content/gdrive/MyDrive')
from sca_IO_func import LinearRegressionModelV2
#Carico il modello
percorso_modello = '/home/scaioli/Datasets/modello_1PGB.pth'
model = LinearRegressionModelV2('ReLU',1024,4,1485,1,device,0.125)
model.load_state_dict(torch.load(percorso_modello,map_location=torch.device('cpu')))
model.eval()

def tensor_to_dataframe_2(tensore):
    # Assicurati che il tensore abbia le dimensioni 1486 x 1 (1 è il numero di colonne)
    if tensore.dim() != 1 or tensore.size(0) != 1485:
        raise ValueError("Il tensore deve avere dimensioni 1485 x 1")
    # Estrai i dati dal tensore
    dati = tensore.numpy()
    # Crea un DataFrame utilizzando i dati estratti
    df = pd.DataFrame(columns=['Residue Index', 'Residue Index.1', 'Distance (nm)'])
    k=1
    for j in range(55,0, -1):
      for i in range(1,j-1,1):
        nuova_riga = {'Residue Index': i, 'Residue Index.1': j, 'Distance (nm)': tensore[k].item()}
        temp_df = pd.DataFrame([nuova_riga])
        df = pd.concat([df, temp_df], ignore_index=True)
        k+=1
    return df

def dataframe_to_cmap_2(df,time,png_path):
    #colori = ['green','red','white']
    #cmap = ListedColormap(colori)
    # Definisci i limiti delle normali
    #boundaries = [-0.2,0.4,1.2,2]
    #norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    colors = plt.cm.RdYlGn(np.linspace(0, 0.8, 256))
    diverging_colormap = LinearSegmentedColormap.from_list('RdYlGn_diverging', colors, N=256)
    matrice_contatti = np.full((56, 56),2, dtype=float)
    for index, row in df.iterrows():
        i = int(row['Residue Index']) - 1
        j = int(row['Residue Index.1']) - 1
        valore = row['Distance (nm)']
        matrice_contatti[i, j] = valore
        matrice_contatti[j, i] = valore  # La matrice è simmetrica, quindi impostiamo anche l'altro lato
    plt.imshow(matrice_contatti, cmap=diverging_colormap,vmin=0.0,vmax=0.8)
    # Disegna la mappa di contatto# Personalizza gli assi
    plt.xticks(np.arange(1, 55, 5), np.arange(0, 55, 5))
    plt.yticks(np.arange(1, 55, 5), np.arange(0, 55, 5))
    plt.xlabel('Indice del secondo residuo')
    plt.ylabel('Indice del primo residuo')
    plt.title(f'Mappa di contatto al tempo {time}')
    # Aggiungi la barra dei colori
    #plt.colorbar(ticks=[-0.2,0.4,1.2,2], label='distance (nm)', cmap=cmap, boundaries=[-0.2,0.4,1.2,2])
    plt.colorbar()
    # Mostra l'immagine
    plt.savefig(png_path,bbox_inches='tight')
    plt.close()
    return None

def inverse_optimization(model, input_data, output_desired, learning_rate, epochs,lr_decay_threshold):
    # Imposta il modello in modalità di valutazione
    model.eval()
    input_data_tensor = input_data.clone().detach().requires_grad_(True)
    for epoch in range(epochs):
      # Calcola l'output della tua rete
      #output_actual = model(torch.sigmoid(input_data_tensor)*0.8)
      output_actual = model(input_data_tensor)
      # Calcola la perdita inversa
      loss = abs(output_actual-output_desired)
      num_zeros = torch.sum(output_actual == 0.0).float()
      penalty = torch.clamp(num_zeros - 125, min=0.0)
      # La penalità viene aggiunta alla loss
      loss += penalty
      # Aggiunge la penalità per la somma inferiore a 1088 (0.8*(1485-125)) quando il mio tensore è completamente legato ha un valore di 1088 e questo valore è un lim inferiore 
      sum_penalty = torch.clamp(torch.sum(output_actual) - 1088, min=0.0)*4
      loss += sum_penalty
      # Definisci l'ottimizzatore
      optimizer = optim.SGD([input_data_tensor], lr=learning_rate)
      # Azzera i gradienti
      optimizer.zero_grad()
      # Calcola i gradienti
      loss.backward()
      # Aggiorna l'input utilizzando l'ottimizzatore
      optimizer.step()

      # Visualizzazione della perdita ogni tot iterazioni
      if epoch % 10000 == 0:
          #print(f'Epoch {epoch}, Loss: {loss.item()}')
          print(f"loss = {loss.item()}")
      # Ottieni l'input invertito
      inverted_input = input_data_tensor.detach().clone()
    return inverted_input

def crea_cartella(nome_cartella):
    try:
        os.makedirs(nome_cartella)
        print(f"Cartella '{nome_cartella}' creata con successo.")
    except FileExistsError:
        print(f"La cartella '{nome_cartella}' esiste già.")


times=np.array([0.1])
learning_rate=0.001
lr_decay_threshold=2
epochs = 500000

for time in times:
  directory_path=f'/home/scaioli/Risultati_Montecarlo/tempo_{time}'
  crea_cartella(directory_path)
  tensor_directory_path=os.path.join(directory_path,f'/tensori_tempo_{time}')
  png_directory_path=os.path.join(directory_path,f'/immagini_tempo_{time}')
  crea_cartella(tensor_directory_path)
  crea_cartella(png_directory_path)
  inverted_input=torch.ones(1485)*0.8
  #inverted_input=torch.load('/content/gdrive/MyDrive/tensore_tempo_pred_0.98')
  #inverted_input = torch.from_numpy(np.random.choice([0.0, 0.8], size=1485, replace=True).astype(np.float32))
  for i in range(1,500):
    print(f"         Iterazione numero {i} :")
    print(f"_____________________________________________")
    inverted_input = inverse_optimization(model=model, input_data=inverted_input, output_desired=time,learning_rate=learning_rate,epochs=epochs,lr_decay_threshold=lr_decay_threshold)
    #inverted_input = torch.where(inverted_input > 0.5, torch.tensor(0.8), torch.tensor(0.0))
    time_pred=model(inverted_input)
    time_pred='{:.2f}'.format(time_pred.item())
    print(f'tempo predetto: {time_pred}')
    png_path=os.path.join(png_directory_path,f'/cmap_inverted_time_{time_pred}.png')
    inverted_dataframe = tensor_to_dataframe_2(inverted_input)
    inverted_cmap = dataframe_to_cmap_2(inverted_dataframe,time_pred,png_path)
    tensor_path=os.path.join(tensor_directory_path,f'/tensore_tempo_pred_{time_pred}.pth')
    torch.save(inverted_input,tensor_path)