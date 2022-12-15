#!/bin/bash

NN=4
infile=antarctica_ice_velocity_450m_cf.nc

for grid in 16000 8000 4000 2000 1000 500; do
    if [ ! -f 2sar_weights_${grid}m.nc ]; then
        echo "Preparing weights for ${grid}m grid"
        cdo -f nc4 -z zip_2 -P $NN genycon,../grids/g${grid}m.txt $infile 2sar_weights_${grid}m.nc
    fi
done


for grid in 16000 8000 4000 2000 1000 500; do
    echo "Preparing ${grid}m grid"
    outfile=antarctica_ice_velocity_g${grid}m
    cdo -O -L -f nc4 -z zip_2 -P $NN setctomiss,0 -remap,../grids/g${grid}m.txt,sar_weights_${grid}m.nc -chname,magnitude,velsurf_mag -selvar,magnitude $infile ${outfile}.nc
    python add_coord.py ${outfile}.nc ../bed_dem/pism_BedMachineBedmapHFAntarctica_v01_g${grid}m.nc mask 2,4 -o ${outfile}_no_shelves.nc
done
