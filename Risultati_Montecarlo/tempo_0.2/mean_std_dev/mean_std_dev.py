import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import glob

def tensor_to_cmap(tensore,png_path,time):
    if tensore.dim() != 1 or tensore.size(0) != 1485:
        raise ValueError("Il tensore deve avere dimensioni 1485 x 1")
    # Estrai i dati dal tensore
    dati = tensore.numpy()
    media = np.mean(dati)
    print(f'media di appoggio = {media}')
    # Crea un DataFrame utilizzando i dati estratti
    df = pd.DataFrame(columns=['Residue Index', 'Residue Index.1', 'Distance (nm)'])
    k=1
    for j in range(55,0, -1):
      for i in range(1,j-1,1):
        nuova_riga = {'Residue Index': i, 'Residue Index.1': j, 'Distance (nm)': tensore[k].item()}
        temp_df = pd.DataFrame([nuova_riga])
        df = pd.concat([df, temp_df], ignore_index=True)
        k+=1
    #colors = plt.cm.RdYlGn(np.linspace(0,0.8, 256))
    #diverging_colormap = LinearSegmentedColormap.from_list('RdYlGn_diverging', colors, N=256)
    matrice_contatti = np.full((55, 55),1, dtype=float)
    for index, row in df.iterrows():
        i = int(row['Residue Index']) - 1
        j = int(row['Residue Index.1']) - 1
        valore = row['Distance (nm)']
        matrice_contatti[i, j] = valore
        matrice_contatti[j, i] = valore  # La matrice è simmetrica, quindi impostiamo anche l'altro lato
    matrice_contatti[matrice_contatti==1]=media
    print(f'matrice contatti = {matrice_contatti}')
    colors = plt.cm.RdYlGn(np.linspace(0,0.8, 256))
    diverging_colormap = LinearSegmentedColormap.from_list('RdYlGn_diverging', colors, N=256)
    plt.imshow(matrice_contatti, cmap=diverging_colormap,vmin=matrice_contatti.min(),vmax=matrice_contatti.max())
    # Disegna la mappa di contatto# Personalizza gli assi
    plt.xticks(np.arange(0, 55, 5), np.arange(0, 55, 5))
    plt.yticks(np.arange(0, 55, 5), np.arange(0, 55, 5))
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

def carica_tensori_da_cartella(cartella):
    # Trova tutti i file con estensione .pth nella cartella
    percorsi_tensori = glob.glob(os.path.join(cartella, '*.pth'))
    # Carica i tensori da ciascun file
    tensori = [torch.load(percorso) for percorso in percorsi_tensori]
    return tensori

def somma_e_normalizza_tensori(tensori):
    # Somma tutti i tensori
    somma_tensori = torch.stack(tensori).sum(dim=0)
    # Normalizza la somma
    somma_normalizzata = somma_tensori / len(tensori)
    return somma_normalizzata

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
    
    time=0.2
    # Specifica il percorso della cartella contenente i tensori .pth
    percorso_cartella = f"/home/scaioli/Risultati_Montecarlo/tempo_{time}/tensori_tempo_{time}"
    
    # Carica i tensori dalla cartella
    tensori = carica_tensori_da_cartella(percorso_cartella)

    if tensori:
        # Somma e normalizza i tensori
        somma_normalizzata = somma_e_normalizza_tensori(tensori)

        # Calcola media e deviazione standard dei tensori
        media, deviazione_standard = calcola_media_e_deviazione_standard(tensori)

        print("Somma e normalizzazione dei tensori:")
        print(somma_normalizzata)
        tensor_to_cmap(tensore=somma_normalizzata,png_path=f'/home/scaioli/Risultati_Montecarlo/tempo_{time}/mean_std_dev/somma_norm.png',time=time)

        print("\nMedia dei tensori:")
        print(media)
        tensor_to_cmap(tensore=media,png_path=f'/home/scaioli/Risultati_Montecarlo/tempo_{time}/mean_std_dev/media.png',time=time)
        
        print("\nDeviazione standard dei tensori:")
        print(deviazione_standard)
        tensor_to_cmap(tensore=deviazione_standard,png_path=f'/home/scaioli/Risultati_Montecarlo/tempo_{time}/mean_std_dev/std_dev.png',time=time)

    else:
        print("Nessun file .pth trovato nella cartella.")


