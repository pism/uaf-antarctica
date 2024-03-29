#!/usr/bin/env python3
# (C) 2017-2021 Andy Aschwanden, Doug Brinkerhoff

# This script draws samples with the Sobol Sequences
# using the Saltelli method
#
# Herman, J., Usher, W., (2017), SALib:
# An open-source Python library for Sensitivity Analysis, Journal of Open Source Software,
# 2(9), 97, doi:10.21105/joss.00097

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.stats.distributions import uniform

from SALib.sample import saltelli
from pyDOE import lhs

parser = ArgumentParser()
parser.description = "Draw samples using the Saltelli methods"
parser.add_argument(
    "-s", "--n_samples", dest="n_samples", type=int, help="""number of samples to draw. default=10.""", default=10
)
parser.add_argument(
    "--sliding_law",
    choices=["pseudo_plastic", "regularized_coulomb"],
    help="""Sliding law.""",
    default="pseudo_plastic",
)
parser.add_argument(
    "-m",
    "--method",
    dest="method",
    type=str,
    choices=["lhs", "saltelli"],
    help="""number of samples to draw. default=saltelli.""",
    default="lhs",
)
parser.add_argument("OUTFILE", nargs=1, help="Ouput file (CSV)", default="samples.csv")
options = parser.parse_args()
method = options.method
n_samples = options.n_samples
outfile = options.OUTFILE[-1]
sliding_law = options.sliding_law


distributions = {
    "sia_e": uniform(loc=0.5, scale=3.5),  # uniform between 1 and 4
    "ssa_e": uniform(loc=0.5, scale=1.5),  # uniform between 1 and 2
    "sia_n": uniform(loc=3.0, scale=1.0),  # uniform between 1 and 2
    "ssa_n": uniform(loc=3.0, scale=0.5),  # uniform between 1 and 2
    "ppq": uniform(loc=0.25, scale=0.7),  # uniform between 0.25 and 0.95
    "tefo": uniform(loc=0.015, scale=0.045),  # uniform between 0.015 and 0.050
    "phi_min": uniform(loc=10.0, scale=20.0),  # uniform between  10 and 30
    "z_min": uniform(loc=-1000, scale=1000),  # uniform between -1000 and 0
    "z_max": uniform(loc=0, scale=1000),  # uniform between 0 and 1000
    "pseudo_plastic_uthreshold": uniform(loc=0, scale=200),
}


# Names of all the variables
keys = [x for x in distributions.keys()]

# Describe the Problem
problem = {"num_vars": len(keys), "names": keys, "bounds": [[0, 1]] * len(keys)}

# Generate uniform samples (i.e. one unit hypercube)
if method == "saltelli":
    unif_sample = saltelli.sample(problem, n_samples, calc_second_order=False)
elif method == "lhs":
    unif_sample = lhs(len(keys), n_samples)
else:
    print(f"Method {method} not available")

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

df["sliding_law"] = sliding_law
df["eigen_K"] = 1e13

df.to_csv(f"ensemble_{outfile}", index=True, index_label="id")
