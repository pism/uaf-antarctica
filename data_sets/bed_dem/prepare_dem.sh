#!/bin/bash

set -x
set -e

export HDF5_USE_FILE_LOCKING=FALSE

NN=8  # default number of processors
if [ $# -gt 0 ] ; then
  NN="$1"
fi
N=$NN

bedmap=bedmap2_tiff
wget -nc https://secure.antarctica.ac.uk/data/bedmap2/${bedmap}.zip
unzip -u ${bedmap}.zip

gdal_translate -a_srs epsg:3031 bedrock.tif bedrock.nc
gdal_translate bedmap2_tiff/bedmap2_bed.tif bedmap2_bed.nc


infile=BedMachineAntarctica_2019-11-05_v01.nc
for grid in 16000 8000 4000 2000 1000 500; do
    if [ ! -f machine_weights_${grid}m.nc ]; then
        echo "Preparing weights for ${grid}m grid"
        cdo -f nc4 -z zip_2 -P $NN genycon,../grids/g${grid}m.txt $infile machine_weights_${grid}m.nc
    fi
    if [ ! -f nn_machine_weights_${grid}m.nc ]; then
        echo "Preparing weights for ${grid}m grid"
        cdo -f nc4 -z zip_2 -P $NN gennn,../grids/g${grid}m.txt $infile nn_machine_weights_${grid}m.nc
    fi
    if [ ! -f map_weights_${grid}m.nc ]; then
        echo "Preparing weights for ${grid}m grid"
        cdo -f nc4 -z zip_2 -P $NN genycon,../grids/g${grid}m.txt bedmap2_bed.nc map_weights_${grid}m.nc
    fi
    if [ ! -f pen_weights_${grid}m.nc ]; then
        echo "Preparing weights for ${grid}m grid"
        cdo -f nc4 -z zip_2 -P $NN genycon,../grids/g${grid}m.txt bedrock.nc pen_weights_${grid}m.nc
    fi
done


for grid in 16000 8000 4000 2000 1000 500; do
    echo "Preparing ${grid}m grid"
    outfile=pism_BedMachineAntarctica_v01_g${grid}m.nc
    cdo -O -f nc4 -z zip_2 -P $NN remap,../grids/g${grid}m.txt,machine_weights_${grid}m.nc -delname,mask,source $infile float.nc   # float vars
    cdo -O -f nc4 -z zip_2 -P $NN remap,../grids/g${grid}m.txt,nn_machine_weights_${grid}m.nc -selname,mask,source $infile byte.nc # byte vars
    cdo -O merge float.nc byte.nc $outfile
    rm byte.nc float.nc
    # Make FTT mask: 1 where there is no floating (3) or grounded (2) ice
    ncap2 -O -s "where(thickness<0) thickness=0; ftt_mask[\$y,\$x]=0b; where(mask==0) {thickness=0.; surface=0.;}; where(mask!=2) ftt_mask=1; where(mask!=3) ftt_mask=1;" $outfile $outfile
    ncap2 -O -s 'land_ice_area_fraction_retreat = thickness; where(thickness > 0 || thickness + bed >= (1 - 910.0/1028.0) * thickness + 0) land_ice_area_fraction_retreat = 1;land_ice_area_fraction_retreat@units="1";land_ice_area_fraction_retreat@long_name="maximum ice extent mask";land_ice_area_fraction_retreat@standard_name="";' $outfile $outfile
    ncks -O -4 -d x,-2711000.,-1760000. -d y,800000.,1760000. $outfile pism_BedMachinePeninsula_v01_g${grid}m.nc
    cdo -f nc4 -z zip_2 -P $NN chname,Band1,bed_hf -remap,../grids/g${grid}m.txt,pen_weights_${grid}m.nc bedrock.nc pism_HFAntarctica_g${grid}m.nc
    ncatted -a _FillValue,bed_hf,d,, -a missing_value,bed_hf,d,, pism_HFAntarctica_g${grid}m.nc
    cp pism_BedMachineAntarctica_v01_g${grid}m.nc pism_BedMachineHFAntarctica_v01_g${grid}m.nc
    ncks -A -v bed_hf  pism_HFAntarctica_g${grid}m.nc  pism_BedMachineHFAntarctica_v01_g${grid}m.nc
    ncap2 -O -s "where(bed_hf!=-9999) {bed=bed_hf; thickness=(surface-bed_hf);}; where(thickness<0) {thickness=0;};" pism_BedMachineHFAntarctica_v01_g${grid}m.nc pism_BedMachineHFAntarctica_v01_g${grid}m.nc
    ncks -O -v bed_hf -x  pism_BedMachineHFAntarctica_v01_g${grid}m.nc  pism_BedMachineHFAntarctica_v01_g${grid}m.nc
    cp pism_BedMachineHFAntarctica_v01_g${grid}m.nc  pism_BedMachineBedmapHFAntarctica_v01_g${grid}m.nc
    cdo -f nc4 -z zip_2 -P $NN remap,../grids/g${grid}m.txt,map_weights_${grid}m.nc bedmap2_bed.nc bedmap2_bed_g${grid}m.nc
    ncks -A -v Band1  bedmap2_bed_g${grid}m.nc pism_BedMachineBedmapHFAntarctica_v01_g${grid}m.nc
    ncatted -a _FillValue,Band1,d,, -a missing_value,Band1,d,, pism_BedMachineBedmapHFAntarctica_v01_g${grid}m.nc
    ncap2 -O -s "where((mask==0) && (Band1<30000.0)) bed=Band1; where(mask==4) {bed=surface-thickness;};" pism_BedMachineBedmapHFAntarctica_v01_g${grid}m.nc pism_BedMachineBedmapHFAntarctica_v01_g${grid}m.nc
    ncks -O -v Band1 -x pism_BedMachineBedmapHFAntarctica_v01_g${grid}m.nc pism_BedMachineBedmapHFAntarctica_v01_g${grid}m.nc
done
