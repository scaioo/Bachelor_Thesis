import pandas as pd
import numpy as np
import os
import torch
from torch import Tensor
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import re

# Convertire csv a tensore
def csv_to_tensor(csv_file_path: str) -> Tensor:
    # Leggi il file CSV utilizzando pandas
    dataframe = pd.read_csv(csv_file_path)
    # Isola la terza colonna (assumendo che l'indice delle colonne parta da zero)
    terza_colonna = dataframe.iloc[:, 2]
    # Converti la colonna in un tensore PyTorch
    tensor_da_colonna = torch.tensor(terza_colonna.values, dtype=torch.float32)
    return tensor_da_colonna

#Convertire tensore in dataframe
def tensor_to_dataframe(tensore):
    # Assicurati che il tensore abbia le dimensioni 1486 x 1 (1 è il numero di colonne)
    if tensore.dim() != 1 or tensore.size(0) != 1485:
        raise ValueError("Il tensore deve avere dimensioni 1486 x 1")
    # Estrai i dati dal tensore
    dati = tensore.numpy()
    # Crea un DataFrame utilizzando i dati estratti
    df = pd.DataFrame(columns=['Residue Index', 'Residue Index.1', 'Distance (nm)'])
    k=0
    for j in range(56,1, -1):
      for i in range(1,j-1,1):
        nuova_riga = {'Residue Index': i, 'Residue Index.1': j, 'Distance (nm)': tensore[k].item()}
        temp_df = pd.DataFrame([nuova_riga])
        df = pd.concat([df, temp_df], ignore_index=True)
        print(df)
        k+=1
    return df

def dataframe_to_tensor(dataframe):
    # Verifica che il DataFrame abbia tre colonne
    if len(dataframe.columns) != 3:
        raise ValueError("Il DataFrame deve avere esattamente tre colonne.")

    # Seleziona la terza colonna e convertila in un tensore PyTorch
    terza_colonna_tensor = torch.tensor(dataframe.iloc[:, 2].values, dtype=torch.float32)

    return terza_colonna_tensor

def dataframe_to_cmap(df,time,png_path):
    colori = ['white','green']
    cmap = ListedColormap(colori)
    # Definisci i limiti delle normali
    boundaries = [0.,1]
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    matrice_contatti = np.zeros((56, 56))
    for index, row in df.iterrows():
        i = int(row['Residue Index']) - 1
        j = int(row['Residue Index.1']) - 1
        valore = row['Distance (nm)']
        matrice_contatti[i, j] = valore
        matrice_contatti[j, i] = valore  # La matrice è simmetrica, quindi impostiamo anche l'altro lato
    #print(matrice_contatti)
    #plt.imshow(matrice_contatti, cmap=cmap,norm=norm)
    # Disegna la mappa di contatto# Personalizza gli assi
    plt.xticks(np.arange(1, 56, 10), np.arange(1, 56, 10),fontsize='18')
    plt.yticks(np.arange(1, 56, 10), np.arange(1, 56, 10),fontsize='18')
    plt.xlabel('Indice del secondo residuo',fontsize='18')
    plt.ylabel('Indice del primo residuo',fontsize='18')
    plt.title(f'Mappa di contatto Nativa',fontsize='18')
    # Aggiungi la barra dei colori
    #plt.colorbar(ticks=[0,1], label='distance (nm)', cmap=cmap, boundaries=[0,1])
    # Mostra l'immagine
    plt.savefig(png_path,bbox_inches='tight')
    plt.close()
    return None

def extract_time_(s):
    try:
        # Utilizza un'espressione regolare per estrarre la parte numerica dalla stringa
        match = re.search(r'\d+\.\d+', s)
        if match:
            return float(match.group())
        else:
            return None  # Puoi gestire il caso in cui non viene trovata alcuna corrispondenza
    except ValueError:
        return None  # Gestisci il caso in cui la conversione fallisce

def make_cmap_null():
  df = pd.DataFrame(columns=['Residue Index', 'Residue Index.1', 'Distance (nm)'])
  for j in range(56,0, -1):
    for i in range(1,j-1,1):
      nuova_riga = {'Residue Index': i, 'Residue Index.1': j, 'Distance (nm)': 0.8}
      temp_df = pd.DataFrame([nuova_riga])
      df = pd.concat([df, temp_df], ignore_index=True)
  return df

def make_beta_hairpin_down(dataframe):
  condizioni =((dataframe['Residue Index'] == 46) & (dataframe['Residue Index.1'] == 48)) | \
              ((dataframe['Residue Index'] == 47) & (dataframe['Residue Index.1'] == 49)) | \
              ((dataframe['Residue Index'] == 46) & (dataframe['Residue Index.1'] == 50)) | \
              ((dataframe['Residue Index'] == 45) & (dataframe['Residue Index.1'] == 51)) | \
              ((dataframe['Residue Index'] == 44) & (dataframe['Residue Index.1'] == 52)) | \
              ((dataframe['Residue Index'] == 43) & (dataframe['Residue Index.1'] == 53)) | \
              ((dataframe['Residue Index'] == 42) & (dataframe['Residue Index.1'] == 54)) | \
              ((dataframe['Residue Index'] == 41) & (dataframe['Residue Index.1'] == 55)) | \
              ((dataframe['Residue Index'] == 40) & (dataframe['Residue Index.1'] == 56)) | \
              ((dataframe['Residue Index'] == 48) & (dataframe['Residue Index.1'] == 50))
  dataframe.loc[condizioni, 'Distance (nm)'] = 0.0
  return dataframe

def make_beta_hairpin_up(dataframe):
  condizioni =((dataframe['Residue Index'] == 8) & (dataframe['Residue Index.1'] == 10)) | \
              ((dataframe['Residue Index'] == 10) & (dataframe['Residue Index.1'] == 12)) | \
              ((dataframe['Residue Index'] == 9) & (dataframe['Residue Index.1'] == 11)) | \
              ((dataframe['Residue Index'] == 8) & (dataframe['Residue Index.1'] == 12)) | \
              ((dataframe['Residue Index'] == 7) & (dataframe['Residue Index.1'] == 13)) | \
              ((dataframe['Residue Index'] == 6) & (dataframe['Residue Index.1'] == 14)) | \
              ((dataframe['Residue Index'] == 5) & (dataframe['Residue Index.1'] == 15)) | \
              ((dataframe['Residue Index'] == 4) & (dataframe['Residue Index.1'] == 16)) | \
              ((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 17)) | \
              ((dataframe['Residue Index'] == 2) & (dataframe['Residue Index.1'] == 18)) | \
              ((dataframe['Residue Index'] == 1) & (dataframe['Residue Index.1'] == 19))
  dataframe.loc[condizioni, 'Distance (nm)'] = 0.0
  return dataframe

def make_alpha_helix(dataframe):
  condizioni =((dataframe['Residue Index'] == 22) & (dataframe['Residue Index.1'] == 24)) | \
              ((dataframe['Residue Index'] == 23) & (dataframe['Residue Index.1'] == 25)) | \
              ((dataframe['Residue Index'] == 24) & (dataframe['Residue Index.1'] == 26)) | \
              ((dataframe['Residue Index'] == 25) & (dataframe['Residue Index.1'] == 27)) | \
              ((dataframe['Residue Index'] == 26) & (dataframe['Residue Index.1'] == 28)) | \
              ((dataframe['Residue Index'] == 27) & (dataframe['Residue Index.1'] == 29)) | \
              ((dataframe['Residue Index'] == 28) & (dataframe['Residue Index.1'] == 30)) | \
              ((dataframe['Residue Index'] == 29) & (dataframe['Residue Index.1'] == 31)) | \
              ((dataframe['Residue Index'] == 30) & (dataframe['Residue Index.1'] == 32)) | \
              ((dataframe['Residue Index'] == 31) & (dataframe['Residue Index.1'] == 33)) | \
              ((dataframe['Residue Index'] == 32) & (dataframe['Residue Index.1'] == 34)) | \
              ((dataframe['Residue Index'] == 33) & (dataframe['Residue Index.1'] == 35)) | \
              ((dataframe['Residue Index'] == 22) & (dataframe['Residue Index.1'] == 25)) | \
              ((dataframe['Residue Index'] == 23) & (dataframe['Residue Index.1'] == 26)) | \
              ((dataframe['Residue Index'] == 24) & (dataframe['Residue Index.1'] == 27)) | \
              ((dataframe['Residue Index'] == 25) & (dataframe['Residue Index.1'] == 28)) | \
              ((dataframe['Residue Index'] == 26) & (dataframe['Residue Index.1'] == 29)) | \
              ((dataframe['Residue Index'] == 27) & (dataframe['Residue Index.1'] == 30)) | \
              ((dataframe['Residue Index'] == 28) & (dataframe['Residue Index.1'] == 31)) | \
              ((dataframe['Residue Index'] == 29) & (dataframe['Residue Index.1'] == 32)) | \
              ((dataframe['Residue Index'] == 30) & (dataframe['Residue Index.1'] == 33)) | \
              ((dataframe['Residue Index'] == 31) & (dataframe['Residue Index.1'] == 34)) | \
              ((dataframe['Residue Index'] == 32) & (dataframe['Residue Index.1'] == 35)) | \
              ((dataframe['Residue Index'] == 33) & (dataframe['Residue Index.1'] == 36)) | \
              ((dataframe['Residue Index'] == 22) & (dataframe['Residue Index.1'] == 26)) | \
              ((dataframe['Residue Index'] == 23) & (dataframe['Residue Index.1'] == 27)) | \
              ((dataframe['Residue Index'] == 24) & (dataframe['Residue Index.1'] == 28)) | \
              ((dataframe['Residue Index'] == 25) & (dataframe['Residue Index.1'] == 29)) | \
              ((dataframe['Residue Index'] == 26) & (dataframe['Residue Index.1'] == 30)) | \
              ((dataframe['Residue Index'] == 27) & (dataframe['Residue Index.1'] == 31)) | \
              ((dataframe['Residue Index'] == 28) & (dataframe['Residue Index.1'] == 32)) | \
              ((dataframe['Residue Index'] == 29) & (dataframe['Residue Index.1'] == 33)) | \
              ((dataframe['Residue Index'] == 30) & (dataframe['Residue Index.1'] == 34)) | \
              ((dataframe['Residue Index'] == 31) & (dataframe['Residue Index.1'] == 35)) | \
              ((dataframe['Residue Index'] == 32) & (dataframe['Residue Index.1'] == 36)) | \
              ((dataframe['Residue Index'] == 33) & (dataframe['Residue Index.1'] == 37))
  dataframe.loc[condizioni, 'Distance (nm)'] = 0.0
  return dataframe

def aggiungi_rumore(dataframe, probabilita_rumore):
    # Crea una copia del DataFrame per non modificare l'originale
    df = dataframe.copy()
    # Itera attraverso le righe del DataFrame
    for index, row in df.iterrows():
            # Controlla se le terze colonne sono 0.0 e la distanza tra le prime due colonne è 1
            if row['Distance (nm)'] == 0.0:
              righe_da_modificare_più = df[(df['Residue Index'] == row['Residue Index']+1) & (df['Residue Index.1'] == row['Residue Index.1'])]
              if np.random.rand() < probabilita_rumore:
                df.loc[righe_da_modificare_più.index, 'Distance (nm)'] = 0.0
              righe_da_modificare_meno = df[(df['Residue Index'] == row['Residue Index']-1) & (df['Residue Index.1'] == row['Residue Index.1'])]
              if np.random.rand() < probabilita_rumore:
                df.loc[righe_da_modificare_meno.index, 'Distance (nm)'] = 0.0
              righe2_da_modificare_più = df[(df['Residue Index.1'] == row['Residue Index.1']+1) & (df['Residue Index'] == row['Residue Index'])]
              if np.random.rand() < probabilita_rumore:
                df.loc[righe2_da_modificare_più.index, 'Distance (nm)'] = 0.0
              righe2_da_modificare_meno = df[(df['Residue Index.1'] == row['Residue Index.1']-1) & (df['Residue Index'] == row['Residue Index'])]
              if np.random.rand() < probabilita_rumore:
                df.loc[righe2_da_modificare_meno.index, 'Distance (nm)'] = 0.0
    return df

def make_rect_down(dataframe):
  condizioni =((dataframe['Residue Index'] == 23) & (dataframe['Residue Index.1'] == 52)) | \
              ((dataframe['Residue Index'] == 27) & (dataframe['Residue Index.1'] == 52)) | \
              ((dataframe['Residue Index'] == 30) & (dataframe['Residue Index.1'] == 52)) | \
              ((dataframe['Residue Index'] == 23) & (dataframe['Residue Index.1'] == 45)) | \
              ((dataframe['Residue Index'] == 30) & (dataframe['Residue Index.1'] == 43)) | \
              ((dataframe['Residue Index'] == 31) & (dataframe['Residue Index.1'] == 43)) | \
              ((dataframe['Residue Index'] == 34) & (dataframe['Residue Index.1'] == 43)) | \
              ((dataframe['Residue Index'] == 41) & (dataframe['Residue Index.1'] == 43))
  dataframe.loc[condizioni, 'Distance (nm)'] = 0.0
  return dataframe

def make_rect_up(dataframe):
  condizioni =((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 30)) | \
              ((dataframe['Residue Index'] == 4) & (dataframe['Residue Index.1'] == 30)) | \
              ((dataframe['Residue Index'] == 5) & (dataframe['Residue Index.1'] == 30)) | \
              ((dataframe['Residue Index'] == 17) & (dataframe['Residue Index.1'] == 32)) | \
              ((dataframe['Residue Index'] == 18) & (dataframe['Residue Index.1'] == 32)) | \
              ((dataframe['Residue Index'] == 18) & (dataframe['Residue Index.1'] == 20)) | \
              ((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 26)) | \
              ((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 23)) | \
              ((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 22)) | \
              ((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 20)) | \
              ((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 21))
  dataframe.loc[condizioni, 'Distance (nm)'] = 0.0
  return dataframe

def make_sbiscia(dataframe):
  condizioni =((dataframe['Residue Index'] == 20) & (dataframe['Residue Index.1'] == 22)) | \
              ((dataframe['Residue Index'] == 20) & (dataframe['Residue Index.1'] == 26)) | \
              ((dataframe['Residue Index'] == 12) & (dataframe['Residue Index.1'] == 37)) | \
              ((dataframe['Residue Index'] == 5) & (dataframe['Residue Index.1'] == 43)) | \
              ((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 45))
  dataframe.loc[condizioni, 'Distance (nm)'] = 0.0
  return dataframe

def make_contact_hairpin(dataframe):
  condizioni =((dataframe['Residue Index'] == 10) & (dataframe['Residue Index.1'] == 56)) | \
              ((dataframe['Residue Index'] == 9) & (dataframe['Residue Index.1'] == 55)) | \
              ((dataframe['Residue Index'] == 8) & (dataframe['Residue Index.1'] == 54)) | \
              ((dataframe['Residue Index'] == 7) & (dataframe['Residue Index.1'] == 53)) | \
              ((dataframe['Residue Index'] == 6) & (dataframe['Residue Index.1'] == 52)) | \
              ((dataframe['Residue Index'] == 5) & (dataframe['Residue Index.1'] == 51)) | \
              ((dataframe['Residue Index'] == 4) & (dataframe['Residue Index.1'] == 50)) | \
              ((dataframe['Residue Index'] == 3) & (dataframe['Residue Index.1'] == 49)) | \
              ((dataframe['Residue Index'] == 2) & (dataframe['Residue Index.1'] == 48)) | \
              ((dataframe['Residue Index'] == 1) & (dataframe['Residue Index.1'] == 47))
  dataframe.loc[condizioni, 'Distance (nm)'] = 0.0
  return dataframe
