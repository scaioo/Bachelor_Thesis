import subprocess

#ciclo per prendere 100 configurazioni iniziali diverse dal tempo 500 al 600
time=500
for i in range(1, 101):
	# Esecuzione di grompp per preparare le simulazioni .tpr
	subprocess.run(['gmx', 'trjconv', '-s','run.tpr','-f', 'traj_comp.xtc', '-o', f'confin_{i}.gro', '-b', f'{i+time}','-e',f'{i+time}','-pbc','mol'])
