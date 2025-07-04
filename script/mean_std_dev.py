import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import glob
from google.colab import drive
drive.mount('/content/gdrive')
def modifica_matrice(matrice,media_tot):
    righe = len(matrice)
    colonne = len(matrice[0])

    for i in range(righe):
        for j in range(colonne):
            if matrice[i][j] > (0.1):
                matrice[i][j] = 1
            else:
                matrice[i][j] = 0
def modifica_matrice_1(matrice,contatti):
    array_matrice = np.array(matrice).flatten()
    indici_massimi = np.argsort(array_matrice)[-contatti:]
    array_matrice[:] = 0
    array_matrice[indici_massimi] = 1
    matrice[:] = np.reshape(array_matrice, matrice.shape)

def tensor_to_cmap_media(tensore,png_path,time,contatti,norm):
    if tensore.dim() != 1 or tensore.size(0) != 1485:
        raise ValueError("Il tensore deve avere dimensioni 1485 x 1")
    # Estrai i dati dal tensore
    dati = tensore.numpy()
    media = np.mean(dati)
    # Crea un DataFrame utilizzando i dati estratti
    df = pd.DataFrame(columns=['Residue Index', 'Residue Index.1', 'Distance (nm)'])
    k=0
    for j in range(56,1, -1):
      for i in range(1,j-1,1):
        nuova_riga = {'Residue Index': i, 'Residue Index.1': j, 'Distance (nm)': tensore[k].item()}
        temp_df = pd.DataFrame([nuova_riga])
        df = pd.concat([df, temp_df], ignore_index=True)
        k+=1
    matrice_contatti = np.full((56, 56),1, dtype=float)
    
    for index, row in df.iterrows():
        i = int(row['Residue Index']) - 1
        j = int(row['Residue Index.1']) - 1
        valore = row['Distance (nm)']
        matrice_contatti[i, j] = valore
        matrice_contatti[j, i] = valore  # La matrice è simmetrica, quindi impostiamo anche l'altro lato
    matrice_contatti[matrice_contatti==1]=media
    if norm==True: modifica_matrice_1(matrice_contatti,contatti)
    colors = plt.cm.Purples(np.linspace(0,1, 256))
    diverging_colormap = LinearSegmentedColormap.from_list('Purples_diverging', colors, N=256)
    plt.imshow(matrice_contatti, cmap=diverging_colormap,vmin=0,vmax=matrice_contatti.max())
    # Disegna la mappa di contatto# Personalizza gli assi
    plt.xticks(np.arange(1, 56, 5), np.arange(1, 56, 5))
    plt.yticks(np.arange(1, 56, 5), np.arange(1, 56, 5))
    plt.xlabel('Indice del secondo residuo')
    plt.ylabel('Indice del primo residuo')
    plt.title(f'Mappa di contatto al tempo {time}')
    # Aggiungi la barra dei colori
    plt.colorbar()
    # Mostra l'immagine
    plt.savefig(png_path,bbox_inches='tight')
    plt.close()
    return None

def tensor_to_cmap_media_nativa(tensore,png_path,time,contatti):
    if tensore.dim() != 1 or tensore.size(0) != 1485:
        raise ValueError("Il tensore deve avere dimensioni 1485 x 1")
    # Estrai i dati dal tensore
    dati = tensore.numpy()
    media = np.mean(dati)
    # Crea un DataFrame utilizzando i dati estratti
    df = pd.DataFrame(columns=['Residue Index', 'Residue Index.1', 'Distance (nm)'])
    k=0
    for j in range(56,1, -1):
      for i in range(1,j-1,1):
        nuova_riga = {'Residue Index': i, 'Residue Index.1': j, 'Distance (nm)': tensore[k].item()}
        temp_df = pd.DataFrame([nuova_riga])
        df = pd.concat([df, temp_df], ignore_index=True)
        k+=1
    matrice_contatti = np.full((56, 56),0, dtype=float)
    for index, row in df.iterrows():
        i = int(row['Residue Index']) - 1
        j = int(row['Residue Index.1']) - 1
        valore = row['Distance (nm)']
        matrice_contatti[i, j] = valore
        matrice_contatti[j, i] = valore  # La matrice è simmetrica, quindi impostiamo anche l'altro lato
    matrice_contatti[matrice_contatti==1]=media
    modifica_matrice_1(matrice_contatti,contatti)
    matrice_nativa=np.loadtxt('/content/gdrive/MyDrive/cmap_native.txt')
    maschera_contatti_uguali = (matrice_contatti == 1) & (matrice_nativa == 1)
    matrice_contatti = matrice_contatti + maschera_contatti_uguali
    matrice_contatti = (matrice_contatti - np.min(matrice_contatti)) / (np.max(matrice_contatti) - np.min(matrice_contatti))
    colori = ['White','Purple','Green']
    valori = [0, 0.5, 1]
    ticks = ['non legato','legato','legato-nativo']
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(valori, colori)))
    plt.imshow(matrice_contatti, cmap=custom_cmap,vmin=0,vmax=1)
    # Disegna la mappa di contatto# Personalizza gli assi
    plt.xticks(np.arange(1, 56, 5), np.arange(1, 56, 5))
    plt.yticks(np.arange(1, 56, 5), np.arange(1, 56, 5))
    plt.xlabel('Indice del secondo residuo')
    plt.ylabel('Indice del primo residuo')
    plt.title(f'Mappa di contatto al tempo {time}')
    # Aggiungi la barra dei colori
    plt.colorbar()
    # Mostra l'immagine
    plt.savefig(png_path,bbox_inches='tight')
    plt.close()
    return None

def tensor_to_cmap_std_dev(tensore,png_path,time):
    if tensore.dim() != 1 or tensore.size(0) != 1485:
        raise ValueError("Il tensore deve avere dimensioni 1485 x 1")
    # Estrai i dati dal tensore
    dati = tensore.numpy()
    media = np.mean(dati)
    #print(f'media di appoggio = {media}')
    # Crea un DataFrame utilizzando i dati estratti
    df = pd.DataFrame(columns=['Residue Index', 'Residue Index.1', 'Distance (nm)'])
    k=0
    for j in range(56,1, -1):
      for i in range(1,j-1,1):
        nuova_riga = {'Residue Index': i, 'Residue Index.1': j, 'Distance (nm)': tensore[k].item()}
        temp_df = pd.DataFrame([nuova_riga])
        df = pd.concat([df, temp_df], ignore_index=True)
        k+=1
    matrice_contatti = np.full((56, 56),1, dtype=float)
    for index, row in df.iterrows():
        i = int(row['Residue Index']) - 1
        j = int(row['Residue Index.1']) - 1
        valore = row['Distance (nm)']
        matrice_contatti[i, j] = valore
        matrice_contatti[j, i] = valore  # La matrice è simmetrica, quindi impostiamo anche l'altro lato
    matrice_contatti[matrice_contatti==1]=media
    #print(f'matrice contatti = {matrice_contatti}')
    colors = plt.cm.Purples(np.linspace(0,1, 256))
    diverging_colormap = LinearSegmentedColormap.from_list('Purples_diverging', colors, N=256)
    plt.imshow(matrice_contatti, cmap=diverging_colormap,vmin=matrice_contatti.min(),vmax=matrice_contatti.max())
    # Disegna la mappa di contatto# Personalizza gli assi
    plt.xticks(np.arange(1, 56, 5), np.arange(1, 56, 5))
    plt.yticks(np.arange(1, 56, 5), np.arange(1, 56, 5))
    plt.xlabel('Indice del secondo residuo')
    plt.ylabel('Indice del primo residuo')
    plt.title(f'Mappa di contatto al tempo {time}')
    plt.colorbar()
    # Mostra l'immagine
    plt.savefig(png_path,bbox_inches='tight')
    plt.close()
    return None

def carica_tensori_da_cartella(cartella):
    # Trova tutti i file con estensione .pth nella cartella
    percorsi_tensori = glob.glob(os.path.join(cartella, '*.pth'))
    # Carica i tensori da ciascun file
    tensori = [torch.load(percorso) for percorso in percorsi_tensori]
    return tensori

def calcola_media_e_deviazione_standard(tensori):
    # Converte i tensori in un array numpy
    array_tensori = torch.stack(tensori).numpy()
    # Calcola la media e la deviazione standard lungo l'asse 0 (tutti i tensori)
    media = np.mean(array_tensori, axis=0)
    deviazione_standard = np.std(array_tensori, axis=0)
    media_t = torch.from_numpy(media)
    deviazione_standard_t = torch.from_numpy(deviazione_standard)
    return media_t, deviazione_standard_t

if __name__ == "__main__":
    times = [0.1,0.2,0.3]
    contacts = [80,85,90]
    for (time,contatti) in zip(times,contacts):
        # Specifica il percorso della cartella contenente i tensori .pth
        percorso_cartella = f"/content/gdrive/MyDrive/Risultati_Montecarlo/tempo_{time}/tensori_norminv_tempo_{time}"

        # Carica i tensori dalla cartella
        tensori = carica_tensori_da_cartella(percorso_cartella)

        if tensori:
            # Calcola media e deviazione standard dei tensori
            media, deviazione_standard = calcola_media_e_deviazione_standard(tensori)

            print("\nMedia dei tensori:")
            print(media)
            tensor_to_cmap_media(tensore=media,png_path=f'/content/gdrive/MyDrive/Risultati_Montecarlo/tempo_{time}/mean_std_dev/media.png',time=time,contatti=contatti,norm=False)
            tensor_to_cmap_media(tensore=media,png_path=f'/content/gdrive/MyDrive/Risultati_Montecarlo/tempo_{time}/mean_std_dev/media_norm.png',time=time,contatti=contatti,norm=True)
            tensor_to_cmap_media_nativa(tensore=media,png_path=f'/content/gdrive/MyDrive/Risultati_Montecarlo/tempo_{time}/mean_std_dev/media_nativa.png',time=time,contatti=contatti)
            print("\nDeviazione standard dei tensori:")
            print(deviazione_standard)
            tensor_to_cmap_std_dev(tensore=deviazione_standard,png_path=f'/content/gdrive/MyDrive/Risultati_Montecarlo/tempo_{time}/mean_std_dev/std_dev.png',time=time)

        else:
            print("Nessun file .pth trovato nella cartella.")
