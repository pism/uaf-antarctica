#!/bin/bash

set -x
set -e

NN=$1

bedmap=bedmap2_tiff
wget -nc https://secure.antarctica.ac.uk/data/bedmap2/${bedmap}.zip
unzip -u ${bedmap}.zip

infile=BedMachineAntarctica_2019-11-05_v01.nc
for grid in 16000 8000 4000 2000 1000 500; do
    echo "Preparing ${grid}m grid"
    outfile=pism_BedMachineAntarctica_v01_g${grid}m.nc
    # cdo -f nc4 -z zip_2 -P $NN remapycon,../grids/g${grid}m.txt $infile $outfile
    # Make FTT mask: 1 where there is no floating (3) or grounded (2) ice
    ncap2 -O -s "where(thickness<0) thickness=0; ftt_mask[\$y,\$x]=0b; where(mask==0) {thickness=0.; surface=0.;}; where(mask!=2) ftt_mask=1; where(mask!=3) ftt_mask=1;" $outfile $outfile
    
done
