import subprocess 
#Create a cicle for each traj, I have previously created 100 trajectory protein folding 
for i in range(1,100):
	#Create a cicle for each contact map depending on time t=[0,500] ps one map each 0.5 ps
	for j in range (0,1001):
		proc = subprocess.Popen(['gmx','mdmat','-f',f'traj_comp_{i}.xtc','-s',f'confin_{i}.gro','-b',f'{j/2}','-e',f'{j/2}','-nlevels','1','-t','0.4','-frames',f'cmap_traj_{i}_time_{j}'],stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
		#,'-t','0.3'
		#'-no',f'cmap_traj_{i}_time_{j/2}',
		proc.stdin.write('1')
		proc.stdin.flush()
		output, error = proc.communicate()
		proc.wait()
		subprocess.run(['rm','dm.xpm'])
		subprocess.run(['dit', 'xpm2csv', '-f', f'cmap_traj_{i}_time_{j}.xpm'])
		subprocess.run(['rm',f'cmap_traj_{i}_time_{j}.xpm'])
