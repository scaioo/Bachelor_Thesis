#!/bin/bash
#ciclo for per le 100 traiettorie
for i in {1..101}; do
#echo "ciclo for per traittoria $i"	
	for j in {0..1001}; do
		LC_NUMERIC=C;
		t=$(echo "scale=2; $j / 2" | bc)
		time=$(printf "%.2f\n" "$(echo "scale=2; $j / 2" | bc)")
		#echo 'eseguo comando gmx' 
		echo "1\n" | gmx mdmat -f traj_comp_${i}.xtc -s simulation_${i}.tpr -b $t -e $t -nlevels 2 -t 0.4 -frames comap_traj
		rm dm.xpm 
		dit xpm2csv -f comap_traj.xpm 
		mv dit_comap_traj.csv cmap_traj_${i}_time_$time.csv
		rm comap_traj.xpm 
	done
done
