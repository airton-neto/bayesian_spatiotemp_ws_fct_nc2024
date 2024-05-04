# %%

from itertools import product
from typing import List, Union

import numpy as np
import pandas as pd

# Para um set de todas as variáveis, use dir(dataset)
# ou busque em https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs20221101/gfs_0p25_1hr_00z
# para o caso do NOAA. atualize o link pra uma data mais recente.
# Atenção: os floats virarão tags no banco como str, até mesmo a quantidade de zeros após a casa decimal influencia

ncar_to_noaa_height_variables = {
    "Dewpoint_temperature_height_above_ground": {2.0: "dpt2m"},
    "Pressure_height_above_ground": {80.0: "pres80m"},
    "Relative_humidity_height_above_ground": {2.0: "rh2m"},
    "Specific_humidity_height_above_ground": {2.0: "spfh2m", 80.0: "spfh80m"},
    "Temperature_height_above_ground": {
        2.0: "tmp2m",
        80.0: "tmp80m",
        100.0: "tmp100m",
    },
    # "U-Component_Storm_Motion_height_above_ground_layer": {
    #     6000.0: "ustm6000_0m"
    # },
    # "V-Component_Storm_Motion_height_above_ground_layer": {
    #     6000.0: "vstm6000_0m"
    # },
    "u-component_of_wind_height_above_ground": {
        10.0: "ugrd10m",
        20.0: "ugrd20m",
        30.0: "ugrd30m",
        40.0: "ugrd40m",
        50.0: "ugrd50m",
        80.0: "ugrd80m",
        100.0: "ugrd100m",
    },
    "v-component_of_wind_height_above_ground": {
        10.0: "vgrd10m",
        20.0: "vgrd20m",
        30.0: "vgrd30m",
        40.0: "vgrd40m",
        50.0: "vgrd50m",
        80.0: "vgrd80m",
        100.0: "vgrd100m",
    },
}

ncar_to_noaa_altitude_variables = {
    "u-component_of_wind_altitude_above_msl": {
        1829.0: "ugrd_1829m",
        2743.0: "ugrd_2743m",
        3658.0: "ugrd_3658m",
    },
    "v-component_of_wind_altitude_above_msl": {
        1829.0: "vgrd_1829m",
        2743.0: "vgrd_2743m",
        3658.0: "vgrd_3658m",
    },
}

ncar_to_noaa_other_variables = {
    "Planetary_Boundary_Layer_Height_surface": "hpblsfc",
    "u-component_of_wind_maximum_wind": "ugrdmwl",
    "v-component_of_wind_maximum_wind": "vgrdmwl",
    "Vertical_Speed_Shear_tropopause": "vwshtrop",
    "u-component_of_wind_tropopause": "ugrdtrop",
    "v-component_of_wind_tropopause": "vgrdtrop",
    "u-component_of_wind_planetary_boundary": "ugrdpbl",
    "v-component_of_wind_planetary_boundary": "vgrdpbl",
    "Geopotential_height_maximum_wind": "hgtmwl",
    "Temperature_maximum_wind": "tmpmwl",
    "u-component_of_wind_sigma": "ugrdsig995",
    "v-component_of_wind_sigma": "vgrdsig995",
    "Wind_speed_gust_surface": "gustsfc",
    "Pressure_surface": "pressfc",
}


# %%
_lines: List[List[Union[str, float, None]]] = []
for ncar, noaa in ncar_to_noaa_other_variables.items():
    _lines.append([ncar, None, None, noaa])

for ncar, dic in ncar_to_noaa_height_variables.items():
    for height, noaa in dic.items():
        _lines.append([ncar, height, None, noaa])

for ncar, dic in ncar_to_noaa_altitude_variables.items():
    for altitude, noaa in dic.items():
        _lines.append([ncar, None, altitude, noaa])

feature_df = pd.DataFrame(
    _lines, columns=["ncar", "height", "altitude", "noaa"]
)
ncar_feature_list = list(pd.unique(feature_df["ncar"]))
