import os
import numpy as np

# Directory contenente i file CSV
directory_path = "/home/scaio/Data_norm_mappe"

# Elenco dei file CSV nella directory
file_list = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

# Per ogni file CSV nella lista
for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        print(f"file path: {file_path}")
        # Carica il file CSV utilizzando NumPy
        try:
            data = np.loadtxt(file_path, delimiter=',', usecols=2, dtype=str)
            # Salva la terza colonna come file CSV
            np.savetxt(file_path, data, fmt='%s', delimiter=',', newline='\n')
            print(f"file {file_path} salvato ")
        except ValueError:
            print(f"Il file '{file_name}' ha meno di tre colonne. Salta questo file.")
            continue   
        # Carica il file CSV utilizzando NumPy
        #data = np.loadtxt(file_path, delimiter=',', usecols=2, dtype=str)            
        #if len(data.shape) != 1:
            # Salva la terza colonna come file CSV
        #    np.savetxt(file_path, data, fmt='%s', delimiter=',', newline='\n')

