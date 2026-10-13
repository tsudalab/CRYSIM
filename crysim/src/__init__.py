import os
from pathlib import Path
from dotenv import load_dotenv
import sys
current_path = os.path.split(__file__)[0]
project_dir = Path(current_path).parents[1]
load_dotenv(f'{str(project_dir)}/.env')
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
assert (
    PROJECT_ROOT is not None
), "Please configure the PROJECT_ROOT in the .env file"
print('init root directory')
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + '/process')
sys.path.append(PROJECT_ROOT + '/script')

REG_ARGS = f"{PROJECT_ROOT}/crysim/src/regressor/reg_param.yml"

DATASET_ROOT = os.environ.get("DATASET_ROOT")
DATASET_ROOT = PROJECT_ROOT if (DATASET_ROOT is None) or (DATASET_ROOT == "") else DATASET_ROOT
DATASET_ROOT += "/dataset"
Path(DATASET_ROOT).mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = os.environ.get("RESULTS_ROOT")
RESULTS_ROOT = PROJECT_ROOT if (RESULTS_ROOT is None) or (RESULTS_ROOT == "") else RESULTS_ROOT
RESULTS_ROOT += "/results"
# Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)

SYM_ROOT = os.environ.get("SYM_ROOT")
SYM_ROOT = PROJECT_ROOT if (SYM_ROOT is None) or (SYM_ROOT == "") else SYM_ROOT
SYM_ROOT += "/sym_data"
Path(SYM_ROOT).mkdir(parents=True, exist_ok=True)

INCAR_ROOT = os.environ.get("INCAR_ROOT")
INCAR_ROOT = PROJECT_ROOT if (INCAR_ROOT is None) or (INCAR_ROOT == "") else INCAR_ROOT
INCAR_ROOT += "/incar"

import torch
DEFAULT_TORCH_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_TORCH_DTYPE)

import numpy as np
DEFAULT_NUMPY_DTYPE = np.float32
# DEFAULT_NUMPY_DTYPE = np.float64

# DEBUG = True
DEBUG = False
