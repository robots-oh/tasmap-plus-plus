from addict import Dict
import os
import torch

macros = Dict()
tp = macros.globals

tp.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### PATH PRESET
tp.ENV_PATH = "/workspace/data/scenes"

tp.OBJ_DATA_PATH = "/workspace/data/obj_data"
tp.CUS_USD_PATH = "/workspace/data/obj_data/custom_obj_usd"
tp.FW_TEXTURE_PATH = "/workspace/data/obj_data/fw_textures"




##### SCENE CREATION VARIABLES
tp.SC_WALL_COLOR = {
    'skyblue' : (0.5527426, 0.9254611, 1),
    'green' : (0.81215376, 0.98312235, 0.61808115),
    'beige' : (0.9957806, 0.95589185, 0.7857003),
    'blue' : (0.29535866, 0.59863085, 1),
    'white' : (0.87, 0.87, 0.87),
    'light_grey' : (0.7, 0.7, 0.7),
    'mint_green': (0.440, 0.705, 0.621),
    'pastel_blue': (0.383, 0.738, 0.734),
    'light_olive': (0.621, 0.705, 0.571),
    'soft_ivory': (0.7, 0.7, 0.5),
    'pale_yellow': (0.840, 0.832, 0.273),
    'soft_grey_blue': (0.401, 0.489, 0.489),
}



