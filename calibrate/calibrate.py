#!/usr/bin/env python
# Copyright (C) 2016-21 Andy Aschwanden

import itertools
from collections import OrderedDict
import csv
import numpy as np
import pandas as pd
import os
import sys
import shlex
from os.path import join, abspath, realpath, dirname

try:
    import subprocess32 as sub
except:
    import subprocess as sub

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys


def current_script_directory():
    import inspect

    filename = inspect.stack(0)[0][1]
    return realpath(dirname(filename))


script_directory = current_script_directory()

sys.path.append(join(script_directory, "../resources"))
from resources import *


def map_dict(val, mdict):
    try:
        return mdict[val]
    except:
        return val


grid_choices = [500, 1000, 2000, 4000, 8000, 16000]

# set up the option parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = "Generating scripts to calibrate Antarctic model."
parser.add_argument("FILE", nargs=1, help="Input file to restart from", default=None)
parser.add_argument(
    "-n",
    "--n_procs",
    dest="n",
    type=int,
    help="""number of cores/processors. default=96.""",
    default=96,
)
parser.add_argument(
    "-w",
    "--wall_time",
    dest="walltime",
    help="""walltime. default: 24:00:00.""",
    default="24:00:00",
)
parser.add_argument(
    "-q",
    "--queue",
    dest="queue",
    choices=list_queues(),
    help="""queue. default=long.""",
    default="t2standard",
)
parser.add_argument(
    "-d",
    "--domain",
    dest="domain",
    choices=["antarctica", "peninsula"],
    help="sets the modeling domain",
    default="antarctica",
)
parser.add_argument(
    "--exstep",
    dest="exstep",
    help="Writing interval for spatial time series",
    default=1,
)
parser.add_argument(
    "-f",
    "--o_format",
    dest="oformat",
    choices=["netcdf3", "netcdf4_parallel", "netcdf4_serial", "pnetcdf"],
    help="output format",
    default="netcdf4_serial",
)
parser.add_argument(
    "-g",
    "--grid",
    dest="grid",
    type=int,
    choices=grid_choices,
    help="horizontal grid resolution",
    default=8000,
)
parser.add_argument(
    "--i_dir",
    dest="input_dir",
    help="input directory",
    default=abspath(join(script_directory, "..")),
)
parser.add_argument("--o_dir", dest="output_dir", help="output directory", default="test_dir")
parser.add_argument(
    "--o_size",
    dest="osize",
    choices=["small", "medium", "big", "big_2d", "custom"],
    help="output size type",
    default="custom",
)
parser.add_argument(
    "-s",
    "--system",
    dest="system",
    choices=list_systems(),
    help="computer system to use.",
    default="chinook",
)
parser.add_argument(
    "--spatial_ts",
    dest="spatial_ts",
    choices=["basic", "standard", "none", "hydro"],
    help="output size type",
    default="basic",
)
parser.add_argument(
    "--hydrology",
    dest="hydrology",
    choices=["null", "diffuse", "routing", "distributed"],
    help="Basal hydrology model.",
    default="diffuse",
)
parser.add_argument(
    "--stable_gl",
    dest="float_kill_calve_near_grounding_line",
    action="store_false",
    help="Stable grounding line",
    default=True,
)
parser.add_argument(
    "--stress_balance",
    dest="stress_balance",
    choices=["sia", "ssa+sia", "ssa", "blatter"],
    help="stress balance solver",
    default="ssa+sia",
)
parser.add_argument(
    "--dataset_version",
    dest="version",
    choices=["01"],
    help="Input data set version",
    default="01",
)
parser.add_argument(
    "--vertical_velocity_approximation",
    dest="vertical_velocity_approximation",
    choices=["centered", "upstream"],
    help="How to approximate vertical velocities",
    default="upstream",
)
parser.add_argument(
    "-e",
    "--ensemble_file",
    dest="ensemble_file",
    help="File that has all combinations for ensemble study",
    default="../uq/ctrl.csv",
)
parser.add_argument(
    "--climate",
    dest="climate",
    help="Climate",
    choices=["flux", "relax"],
    default="flux",
)
parser.add_argument(
    "-L",
    "--comp_level",
    dest="compression_level",
    help="Compression level for output file. Only works with netcdf4_serial.",
    default=2,
)

parser.add_argument("--start_year", dest="start_year", type=int, help="Simulation start year", default=0)
parser.add_argument("--duration", dest="duration", type=int, help="Years to simulate", default=50)
parser.add_argument("--step", dest="step", type=int, help="Step in years for restarting", default=50)

options = parser.parse_args()

nn = options.n
input_dir = abspath(options.input_dir)
output_dir = abspath(options.output_dir)
spatial_tmp_dir = abspath(options.output_dir + "_tmp")

compression_level = options.compression_level
oformat = options.oformat
osize = options.osize
queue = options.queue
walltime = options.walltime
system = options.system

ensemble_file = options.ensemble_file

spatial_ts = options.spatial_ts

climate = options.climate
exstep = options.exstep
float_kill_calve_near_grounding_line = options.float_kill_calve_near_grounding_line
grid = options.grid
hydrology = options.hydrology
stress_balance = options.stress_balance
vertical_velocity_approximation = options.vertical_velocity_approximation
version = options.version

domain = options.domain
pism_exec = generate_domain(domain)

if options.FILE is None:
    print("Missing input file")
    import sys

    sys.exit()
else:
    input_file = options.FILE[0]

pism_dataname = f"$input_dir/data_sets/bed_dem/pism_BedMachineBedmapHFAntarctica_v{version}_g{grid}m.nc"

climate_file = "$input_dir/data_sets/climate/MIROC-ESM-CHEM_2km_clim_1995-2014.nc"

regridvars = "litho_temp,enthalpy,age,tillwat,bmelt,ice_area_specific_volume"
if climate == "relax":
    regridvars += ",thk"

dirs = {"output": "$output_dir", "spatial_tmp": "$spatial_tmp_dir"}
for d in ["performance", "state", "scalar", "spatial", "jobs", "basins"]:
    dirs[d] = "$output_dir/{dir}".format(dir=d)

if spatial_ts == "none":
    del dirs["spatial"]

# use the actual path of the run scripts directory (we need it now and
# not during the simulation)
scripts_dir = join(output_dir, "run_scripts")
if not os.path.isdir(scripts_dir):
    os.makedirs(scripts_dir)

# use the actual path of the experiment table directory (we need it now and
# not during the simulation)
exp_dir = join(output_dir, "experiments")
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

# generate the config file *after* creating the output directory
pism_config = "pism"
pism_config_nc = join(output_dir, pism_config + ".nc")

cmd = "ncgen -o {output} {input_dir}/config/{config}.cdl".format(
    output=pism_config_nc, input_dir=input_dir, config=pism_config
)
sub.call(shlex.split(cmd))

# these Bash commands are added to the beginning of the run scrips
run_header = """# stop if a variable is not defined
set -u
# stop on errors
set -e

# path to the config file
config="{config}"
# path to the input directory (input data sets are contained in this directory)
input_dir="{input_dir}"
# output directory
output_dir="{output_dir}"
# temporary directory for spatial files
spatial_tmp_dir="{spatial_tmp_dir}"

# create required output directories
for each in {dirs};
do
  mkdir -p $each
done

""".format(
    input_dir=input_dir,
    output_dir=output_dir,
    spatial_tmp_dir=spatial_tmp_dir,
    config=pism_config_nc,
    dirs=" ".join(list(dirs.values())),
)

if system != "debug":
    cmd = "lfs setstripe -c -1 {}".format(dirs["output"])
    sub.call(shlex.split(cmd))
    cmd = "lfs setstripe -c -1 {}".format(dirs["spatial_tmp"])
    sub.call(shlex.split(cmd))

# ########################################################
# set up parameters
# ########################################################

tlftw = 0.1

uq_df = pd.read_csv(ensemble_file)
uq_df.fillna(False, inplace=True)

tsstep = "yearly"

scripts = []
scripts_combinded = []

simulation_start_year = options.start_year
simulation_end_year = options.start_year + options.duration
restart_step = options.step

if restart_step > (simulation_end_year - simulation_start_year):
    print("Error:")
    print(
        (
            "restart_step > (simulation_end_year - simulation_start_year): {} > {}".format(
                restart_step, simulation_end_year - simulation_start_year
            )
        )
    )
    print("Try again")
    import sys

    sys.exit(0)

batch_header, batch_system = make_batch_header(system, nn, walltime, queue)

for n, row in enumerate(uq_df.iterrows()):
    combination = row[1]
    print(combination)

    ttphi = f"""{combination["phi_min"]},45,{combination["z_min"]},{combination["z_max"]}"""

    vversion = "v" + str(version)

    name_options = OrderedDict()
    name_options["id"] = combination["id"]

    full_exp_name = "_".join(
        [
            vversion,
            "_".join(["_".join([k, str(v)]) for k, v in list(name_options.items())]),
        ]
    )
    full_outfile = "g{grid}m_{experiment}.nc".format(grid=grid, experiment=full_exp_name)

    # All runs in one script file for coarse grids that fit into max walltime
    script_combined = join(scripts_dir, "cc_g{}m_{}_j.sh".format(grid, full_exp_name))
    with open(script_combined, "w") as f_combined:

        outfiles = []
        job_no = 0
        for start in range(simulation_start_year, simulation_end_year, restart_step):
            job_no += 1

            end = start + restart_step

            experiment = "_".join(
                [
                    vversion,
                    "_".join(["_".join([k, str(v)]) for k, v in list(name_options.items())]),
                    "{}".format(start),
                    "{}".format(end),
                ]
            )

            script = join(scripts_dir, "cc_g{}m_{}.sh".format(grid, experiment))
            scripts.append(script)

            for filename in script:
                try:
                    os.remove(filename)
                except OSError:
                    pass

            if start == simulation_start_year:
                f_combined.write(batch_header)
                f_combined.write(run_header)

            with open(script, "w") as f:

                f.write(batch_header)
                f.write(run_header)

                outfile = "{domain}_g{grid}m_{experiment}.nc".format(
                    domain=domain.lower(), grid=grid, experiment=experiment
                )

                pism = generate_prefix_str(pism_exec)

                general_params_dict = {
                    "profile": join(
                        dirs["performance"],
                        "profile_${job_id}.py".format(**batch_system),
                    ),
                    "ys": start,
                    "ye": end,
                    "calendar": "365_day",
                    "o": join(dirs["state"], outfile),
                    "o_format": oformat,
                    "output.compression_level": compression_level,
                    "config_override": "$config",
                    "front_retreat_file": pism_dataname,
                    "stress_balance.blatter.coarsening_factor": 4,
                    "blatter_Mz": 17,
                    "bp_ksp_type": "gmres",
                    "bp_pc_type": "mg",
                    "bp_pc_mg_levels": 3,
                    "bp_mg_levels_ksp_type": "richardson",
                    "bp_mg_levels_pc_type": "sor",
                    "bp_mg_coarse_ksp_type": "preonly",
                    "bp_mg_coarse_pc_type": "lu",
                    "bp_snes_monitor_ratio": "",
                    "bp_ksp_monitor": "",
                    "stress_balance.ice_free_thickness_standard": 50,
                    "stress_balance.blatter.use_eta_transform": "",
                }

                if start == simulation_start_year:
                    general_params_dict["bootstrap"] = ""
                    general_params_dict["i"] = pism_dataname
                    general_params_dict["regrid_file"] = input_file
                    general_params_dict["regrid_vars"] = regridvars
                else:
                    general_params_dict["i"] = regridfile

                if osize != "custom":
                    general_params_dict["o_size"] = osize
                else:
                    general_params_dict[
                        "output.sizes.medium"
                    ] = "sftgif,velsurf_mag,tempicethk_basal,velsurf,velbase_mag"

                if start == simulation_start_year:
                    grid_params_dict = generate_grid_description(grid, domain)
                else:
                    grid_params_dict = generate_grid_description(grid, domain, restart=True)

                sb_params_dict = {
                    "sia_e": combination["sia_e"],
                    "ssa_e": combination["ssa_e"],
                    "pseudo_plastic_q": combination["ppq"],
                    "till_effective_fraction_overburden": combination["tefo"],
                    "vertical_velocity_approximation": vertical_velocity_approximation,
                    "basal_yield_stress.mohr_coulomb.till_log_factor_transportable_water": tlftw,
                    "basal_resistance.pseudo_plastic.u_threshold": combination["pseudo_plastic_uthreshold"],
                    combination["sliding_law"]: "",
                }

                if start == simulation_start_year:
                    sb_params_dict["topg_to_phi"] = ttphi

                # If stress balance choice is made in file, overwrite command line option
                stress_balance_params_dict = generate_stress_balance(stress_balance, sb_params_dict)

                climate_params_dict = generate_climate(
                    climate,
                    force_to_thickness_file=pism_dataname,
                    surface_given_file=climate_file,
                )

                hydro_params_dict = generate_hydrology(hydrology)

                calving_params_dict = generate_calving("eigen_calving")
                calving_params_dict["calving.eigen_calving.K"] = combination["eigen_K"]

                ocean_params_dict = {
                    "ocean": "pico",
                    "ocean_pico_file": "$input_dir/data_sets/ocean/schmidtko_initmip8km.nc",
                }

                scalar_ts_dict = generate_scalar_ts(
                    outfile,
                    tsstep,
                    start=simulation_start_year,
                    end=simulation_end_year,
                    odir=dirs["scalar"],
                )

                all_params_dict = merge_dicts(
                    general_params_dict,
                    grid_params_dict,
                    stress_balance_params_dict,
                    climate_params_dict,
                    hydro_params_dict,
                    calving_params_dict,
                    ocean_params_dict,
                    scalar_ts_dict,
                )

                if not spatial_ts == "none":
                    exvars = spatial_ts_vars[spatial_ts]
                    spatial_ts_dict = generate_spatial_ts(
                        outfile, exvars, exstep, odir=dirs["spatial_tmp"], split=False
                    )

                    all_params_dict = merge_dicts(all_params_dict, spatial_ts_dict)

                all_params = " \\\n  ".join(["-{} {}".format(k, v) for k, v in list(all_params_dict.items())])

                if system == "debug":
                    redirect = " 2>&1 | tee {jobs}/job_{job_no}.${job_id}"
                else:
                    redirect = " > {jobs}/job_{job_no}.${job_id} 2>&1"

                template = "{mpido} {pism} {params}" + redirect

                context = merge_dicts(
                    batch_system,
                    dirs,
                    {"job_no": job_no, "pism": pism, "params": all_params},
                )
                cmd = template.format(**context)

                f.write(cmd)
                f.write("\n")
                f.write(batch_system.get("footer", ""))

                f_combined.write(cmd)
                f_combined.write("\n\n")

                f_combined.write("\n")
                f_combined.write("\n")
                if not spatial_ts == "none":
                    f_combined.write(
                        "mv {tmpfile} {ofile}\n".format(
                            tmpfile=spatial_ts_dict["extra_file"],
                            ofile=join(dirs["spatial"], "ex_" + outfile),
                        )
                    )
                    f_combined.write("\n")

                regridfile = join(dirs["state"], outfile)
                outfiles.append(outfile)

            f_combined.write(batch_system.get("footer", ""))

        scripts_combinded.append(script_combined)

scripts = uniquify_list(scripts)
scripts_combinded = uniquify_list(scripts_combinded)
print("\n".join([script for script in scripts]))
print("\nwritten\n")
print("\n".join([script for script in scripts_combinded]))
print("\nwritten\n")
