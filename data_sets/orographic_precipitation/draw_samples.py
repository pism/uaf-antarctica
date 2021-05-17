#!/usr/bin/env python
# (C) 2017-2019 Andy Aschwanden, Doug Brinkerhoff

# This script draws samples with the Sobol Sequences
# using the Saltelli method
#
# Herman, J., Usher, W., (2017), SALib:
# An open-source Python library for Sensitivity Analysis, Journal of Open Source Software,
# 2(9), 97, doi:10.21105/joss.00097

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.stats.distributions import truncnorm, gamma, uniform, randint

from SALib.sample import saltelli

parser = ArgumentParser()
parser.description = "Draw samples using the Saltelli methods"
parser.add_argument(
    "-s", "--n_samples", dest="n_samples", type=int, help="""number of samples to draw. default=10.""", default=10
)
parser.add_argument("OUTFILE", nargs=1, help="Ouput file (CSV)", default="ltop_calibration_samples.csv")
options = parser.parse_args()
n_samples = options.n_samples
outfile = options.OUTFILE[-1]


distributions = {
    "wind_speed": uniform(loc=5, scale=25),  # uniform between 5 and 30
    "wind_direction": uniform(loc=200, scale=50),  # uniform between 200 and 250
    "water_vapor_scale_height": uniform(loc=1000, scale=2000),  # uniform between 1000 and 3000
    "conversion_time": uniform(loc=500, scale=2500),  # uniform between 500 and 3000
    "fallout_time": uniform(loc=500, scale=2500),  # uniform between 500 and 3000
    "scale_factor": uniform(loc=0.5, scale=0.5),  # uniform between 0.5 and 1
    "background_precip_post": uniform(loc=0, scale=0.2),  # uniform between 0 and 0.2
}


# Generate the Sobol sequence samples with uniform distributions


# Names of all the variables
keys = [x for x in distributions.keys()]

# Describe the Problem
problem = {"num_vars": len(keys), "names": keys, "bounds": [[0, 1]] * len(keys)}

# Generate uniform samples (i.e. one unit hypercube)
unif_sample = saltelli.sample(problem, n_samples, calc_second_order=False)

# To hold the transformed variables
dist_sample = np.zeros_like(unif_sample)

# Now transform the unit hypercube to the prescribed distributions
# For each variable, transform with the inverse of the CDF (inv(CDF)=ppf)
for i, key in enumerate(keys):
    dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])

# Save to CSV file using Pandas DataFrame and to_csv method
header = keys
# Convert to Pandas dataframe, append column headers, output as csv
df = pd.DataFrame(data=dist_sample, columns=header)


df.to_csv(outfile, index=True, index_label="id")
