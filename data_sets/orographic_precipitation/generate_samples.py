#!/usr/bin/env python3

import itertools
import numpy as np
import PISM

from os.path import join
import os

grid = 1000
odir = "calibration_samples"

if not os.path.isdir(odir):
    os.makedir(odir)

# silence initialization messages
PISM.Context().log.set_threshold(1)

file_name = f"../bed_dem/pism_BedMachinePeninsula_v01_g{grid}m.nc"
ctx = PISM.Context()
grid = PISM.IceGrid.FromFile(ctx.ctx, file_name, ["bed"], PISM.CELL_CORNER)

model = PISM.AtmosphereOrographicPrecipitation(grid, PISM.AtmosphereUniform(grid))
geometry = PISM.Geometry(grid)
geometry.bed_elevation.regrid(file_name, critical=True)
geometry.ice_surface_elevation.regrid(file_name, critical=True)


# compute surface elevation from ice thickness and bed elevation
geometry.ensure_consistency(0)

moist_adiabatic_lapse_rate = -6.5

ensemble_file = "ltop_calibration_samples_400.csv"
combinations = np.genfromtxt(ensemble_file, dtype=None, encoding=None, delimiter=",", skip_header=1)


for n, combination in enumerate(combinations):
    (
        id,
        wind_speed,
        wind_direction,
        water_vapor_scale_height,
        conversion_time,
        fallout_time,
        scale_factor,
        background_precip_post,
    ) = combination
    print(f"Generating sample {id} of {len(combinations)}")

    ctx = PISM.Context()
    grid = PISM.IceGrid.FromFile(ctx.ctx, file_name, ["bed"], PISM.CELL_CORNER)

    model = PISM.AtmosphereOrographicPrecipitation(grid, PISM.AtmosphereUniform(grid))
    geometry = PISM.Geometry(grid)
    geometry.bed_elevation.regrid(file_name, critical=True)
    geometry.ice_surface_elevation.regrid(file_name, critical=True)

    config = PISM.Context().config
    config.set_number("orographic_precipitation.grid_size_factor", 2)
    config.set_number("atmosphere.orographic_precipitation.coriolis_latitude", -90)
    config.set_flag("atmosphere.orographic_precipitation.truncate", True)
    config.set_number("atmosphere.orographic_precipitation.grid_size_factor", 2)
    config.set_number("atmosphere.orographic_precipitation.moist_adiabatic_lapse_rate", moist_adiabatic_lapse_rate)

    config.set_number("atmosphere.orographic_precipitation.wind_speed", wind_speed)
    config.set_number("atmosphere.orographic_precipitation.wind_direction", wind_direction)
    config.set_number("atmosphere.orographic_precipitation.conversion_time", conversion_time)
    config.set_number("atmosphere.orographic_precipitation.water_vapor_scale_height", water_vapor_scale_height)
    config.set_number("atmosphere.orographic_precipitation.fallout_time", fallout_time)
    config.set_number("atmosphere.orographic_precipitation.scale_factor", scale_factor)
    config.set_number("atmosphere.orographic_precipitation.background_precip_post", background_precip_post)

    model.init(geometry)
    model.update(geometry, 0, 1)

    f = PISM.util.prepare_output(join(odir, f"ltop_g{grid}m_calibration_sample_{n}.nc"))
    precip = model.mean_precipitation()
    precip.write(f)
    f.close()
