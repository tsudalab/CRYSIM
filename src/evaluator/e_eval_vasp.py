from ase.optimize import BFGS
import matgl.ext.ase as mea
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar
import yaml

from ase.calculators.vasp import Vasp
from pathlib import Path

from src.registry import EnergyEvaluatorName
from src.evaluator.e_eval import EnergyEvaluator, _fmax, _steps


class EnergyVASP(EnergyEvaluator):
    def __init__(self, incar_file):
        with open("token.yml", "r") as tk_dict:
            params = yaml.safe_load(tk_dict)
        vasp_dire = params["vasp_dire"]
        vasp_pp_dire = params["vasp_pp_dire"]
        npar = params["vasp_npar"]
        mpirun_np = params["vasp_mpirun_np"]
        assert vasp_dire != "-", "Please specify directory of the VASP package"
        assert vasp_pp_dire != "-", "Please specify directory of pseudopotential for the VASP package"
        self.npar = npar
        
        super().__init__(name=EnergyEvaluatorName.vasp(),
                         real_name=EnergyEvaluatorName.vasp(),
                         struct_type='ase')
        self.incar_file = incar_file
        self.calc = Vasp(command=f'mpirun -np {mpirun_np} {vasp_dire}/bin/vasp_std')
        incar_settings = self.ext_calc_settings(incar_file) if incar_file is not None else self.default_calc_settings
        self.calc.set(**incar_settings)
        return

    def set_logger(self, logger):
        self.out_dire = '.' if logger is None else logger.log_dir
        self.out_dire += f'/vaspout'
        return

    @property
    def default_calc_settings(self):
        default_setting = {'xc': 'PBE', 'npar': int(self.npar),
                           'algo': 'Fast', 'lreal': 'Auto', 'prec': 'Accurate',
                           'ediff': 1e-5, 'ediffg': -0.02, 'kspacing': 0.4,
                           'nelm': 60, 'ismear': 0, 'sigma': 0.1, 'ispin': 1}
        return default_setting
    
    @property
    def default_point_e_setting(self):
        return {'isif': 2, 'ibrion': -1, 'nsw': 0, 'encut': 400}
    
    @property
    def default_relax_setting(self):
        return {'isif': 3, 'ibrion': 2, 'nsw': 200, 'encut': 520}

    def ext_calc_settings(self, incar_file_dire):
        incar_dict = Incar.from_file(incar_file_dire)
        return {k.lower(): v for k, v in incar_dict.items()}

    def set_output(self, label):
        out_dire = self.out_dire + f'-{label}'
        Path(out_dire).mkdir(parents=False, exist_ok=False)
        self.calc.set(directory=out_dire)
        return

    def cal_energy(self):
        if self.incar_file is None:
            self.calc.set(**self.default_point_e_setting)
        self.atoms.calc = self.calc
        return self.atoms.get_potential_energy()

    def _cal_relax(self):
        if self.incar_file is None:
            self.calc.set(**self.default_relax_setting)
        self.atoms.calc = self.calc
        # Rattle the atoms to get them out of the minimum energy configuration
        self.atoms.rattle(0.5)
        obs = mea.TrajectoryObserver(self.atoms)
        dyn = BFGS(self.atoms, logfile=None)
        dyn.attach(obs, interval=1)
        dyn.run(fmax=_fmax, steps=_steps)
        obs()
        return {
            "final_structure": AseAtomsAdaptor.get_structure(self.atoms),
            "trajectory": obs,
        }
