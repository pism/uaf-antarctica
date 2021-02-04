#!/bin/bash

rsync --delete -rvu --progress aaschwanden@transfer.ccr.buffalo.edu:"/projects/grid/ghub/ISMIP6/Projections/AIS/Atmosphere_Forcing" .
