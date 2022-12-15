import numpy as np
import pandas as pd
import xarray as xr
import os
from optparse import OptionParser

# This script was written to filter the data in a netcdf file based on the value of a field that is contained in a second netcdf file of the same shape


def main():
    parser = OptionParser()
    parser.add_option("-o", "--ouput", dest="output", help="name of output file", metavar="FILE")
    parser.add_option("-k", "--keep", help="add the desired coordinate permanently", action='store_true',default=False)
    (options,args) = parser.parse_args()
    fn1     = os.path.abspath(args[0])           # File to which changes will be made
    fn2     = os.path.abspath(args[1])           # File from which a coordinate will be read and subsquently added to fn1
    coord   = args[2]                            # Desired Coordinate name that exists in fn2
    allowed = args[3]                            # Allowed values for cordinate
    allowed = np.array(allowed.split(','),float) # Split str of allowed values, create list
                                               
    
    with xr.open_dataset(fn1) as original, xr.open_dataset(fn2) as new:
        try : 
            original[coord] = new[coord]
        except Exception as e:
            print(e)        
        data = original.where(original[coord].isin(allowed))
        if not (options.keep):
            data = data.drop_vars([coord,'crs'])

    data.to_netcdf(path=options.output,format='NETCDF4')    
        
    

    return 0

if __name__ == "__main__":
    main()