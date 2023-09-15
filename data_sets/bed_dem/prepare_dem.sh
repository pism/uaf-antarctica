#!/bin/bash

set -x
set -e

export HDF5_USE_FILE_LOCKING=FALSE

NN=4  # default number of processors
if [ $# -gt 2 ] ; then
  NN="$3"
fi
N=$NN

bedmap=bedmap2_tiff
wget -nc https://secure.antarctica.ac.uk/data/bedmap2/${bedmap}.zip
unzip -u ${bedmap}.zip

# gdal_translate -a_srs epsg:3031 bedrock.tif bedrock.nc
# gdal_translate bedmap2_tiff/bedmap2_bed.tif bedmap2_bed.nc


infile=$1
ver=$2
for grid in 16000 8000 4000 2000 1000 500; do
    if [ ! -f machine_weights_${grid}m.nc ]; then
        echo "Preparing weights for ${grid}m grid"
        cdo -v -f nc4 -z zip_2 -P $NN genycon,../grids/g${grid}m.txt $infile machine_weights_${grid}m.nc
    fi
    if [ ! -f nn_machine_weights_${grid}m.nc ]; then
        echo "Preparing weights for ${grid}m grid"
        cdo -v -f nc4 -z zip_2 -P $NN gennn,../grids/g${grid}m.txt $infile nn_machine_weights_${grid}m.nc
    fi
done


for grid in 16000 8000 4000 2000 1000 500; do
    echo "Preparing ${grid}m grid"
    outfile=pism_BedMachineAntarctica_v$ver_g${grid}m.nc
    cdo -L -O -f nc4 -z zip_2 -P $NN remap,../grids/g${grid}m.txt,machine_weights_${grid}m.nc -delname,mask,source $infile float_g${grid}m.nc   # float vars
    cdo -L -O -f nc4 -z zip_2 -P $NN remap,../grids/g${grid}m.txt,nn_machine_weights_${grid}m.nc -selname,mask,source $infile byte_g${grid}m.nc # byte vars
    cdo -O merge float_g${grid}m.nc byte_g${grid}m.nc $outfile
    rm byte.nc float.nc
    # Make FTT mask: 1 where there is no floating (3) or grounded (2) ice
    ncap2 -O -s "where(thickness<0) thickness=0; ftt_mask[\$y,\$x]=0b; where(mask==0) {thickness=0.; surface=0.;}; where(mask!=2) ftt_mask=1; where(mask!=3) ftt_mask=1;" $outfile $outfile
    ncap2 -O -s 'land_ice_area_fraction_retreat = thickness; where(thickness > 0 || thickness + bed >= (1 - 910.0/1028.0) * thickness + 0) land_ice_area_fraction_retreat = 1;land_ice_area_fraction_retreat@units="1";land_ice_area_fraction_retreat@long_name="maximum ice extent mask";land_ice_area_fraction_retreat@standard_name="";' $outfile $outfile
done
