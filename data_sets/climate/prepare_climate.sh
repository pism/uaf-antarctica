#!/bin/bash

for grid in 4000 2000; do
    grid_km=$((grid / 1000))
    cdo -P 6 -O -f nc4 -z zip_2 remapycon,../grids/g${grid}m.txt -setgrid,../grids/ismip6_g${grid}m.txt -chname,smb_clim,climatic_mass_balance,ts_clim,ice_surface_temp,pr_clim,precipitation -selvar,smb_clim,ts_clim,pr_clim Atmosphere_Forcing/miroc-esm-chem_rcp8.5/Regridded_${grid_km}km/MIROC-ESM-CHEM_${grid_km}km_clim_1995-2014.nc MIROC-ESM-CHEM_Antarctica_${grid}m_clim_1995-2014.nc
    ncks -4 -d x,-2711000.,-1760000. -d y,800000.,1760000.  MIROC-ESM-CHEM_Antarctica_${grid}m_clim_1995-2014.nc  MIROC-ESM-CHEM_Peninsula_${grid}m_clim_1995-2014.nc

done
