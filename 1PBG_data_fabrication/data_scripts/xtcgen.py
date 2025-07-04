import subprocess
#ciclo per generare 5 simulazioni
for i in range(1,101):
	#esecuzione di mdrun per generare le traiettorie .xtc
	subprocess.run(['gmx','mdrun','-s', f'simulation_{i}.tpr','-x', f'traj_comp_{i}.xtc'])
	subprocess.run(['rm' , '-r' , 'traj.trr' , 'ener.edr' , 'md.log' , 'mdout.mdp' , 'state.cpt' , 'confout.gro' ])
