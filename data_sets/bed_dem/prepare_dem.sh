#!/bin/bash

set -x
set -e

NN=$1

bedmap=bedmap2_tiff
wget -nc https://secure.antarctica.ac.uk/data/bedmap2/${bedmap}.zip
gunzip ${bedmap}.zip

infile=BedMachineAntarctica_2019-11-05_v01.nc
for grid in 16000 8000 4000 2000 1000 500; do
    echo "Preparing ${grid}m grid"
    cdo -f nc4 -z zip_2 -P $NN remapycon,../grids/g${grid}m.txt $infile pism_BedMachineAntarctica_v01_g${grid}m.nc
    
done
