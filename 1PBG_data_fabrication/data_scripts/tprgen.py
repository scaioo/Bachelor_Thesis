import subprocess

# Ciclo per generare 100 simulazioni
for i in range(1, 101):
    # Esecuzione di grompp per preparare le simulazioni .tpr
    subprocess.run(['gmx', 'grompp', '-f', 'mdpfile.mdp', '-c', f'confin_{i}.gro', '-p', 'topol.top', '-o', f'simulation_{i}.tpr'])
