import numpy as np
from logzero import logger as log_printer
from pyxtal.lattice import para2matrix

from crysim.src.struct_utils import crystal_system_dict


class LatParam:
    def __init__(self, **kwargs):
        dim = kwargs.get('dim')
        self.dim = dim
        if dim == 3:
            self.num_lat_len = 3
            self.num_lat_ang = 3
        elif dim == 2:
            self.num_lat_len = 3
            self.num_lat_ang = 1
        else:
            raise NotImplementedError
        
        self.len_ubs, self.len_lbs, self.len_splits = [0] * self.num_lat_len, [0] * self.num_lat_len, [1] * self.num_lat_len
        self.ang_ubs, self.ang_lbs, self.ang_splits = [0] * self.num_lat_ang, [0] * self.num_lat_ang, [1] * self.num_lat_ang
        self.len_units, self.ang_units = [0] * self.num_lat_len, [0] * self.num_lat_ang
        
        self.lat_len_prec = self.num_lat_len
        self.lat_ang_prec = self.num_lat_ang
        
        return
    
    @property
    def num_var(self):
        return self.lat_len_prec + self.lat_ang_prec
    
    def encode_latparam(self, lat_param, ub, lb, unit, split):
        return np.array([lat_param])
    
    def decode_latparam(self, lat_param, lb, unit, decoding_fn):
        return lat_param.item()


class LatParamDiscretizer(LatParam):
    def __init__(self, **kwargs):
        dim, len_ubs, len_lbs, len_splits, ang_ubs, ang_lbs, ang_splits = \
            kwargs.get('dim'), kwargs.get('len_ubs'), kwargs.get('len_lbs'), kwargs.get('len_splits'), \
            kwargs.get('ang_ubs'), kwargs.get('ang_lbs'), kwargs.get('ang_splits')
        super().__init__(dim=dim)
        assert len(len_ubs) == len(len_lbs) == len(len_splits) == self.num_lat_len
        assert len(ang_ubs) == len(ang_lbs) == len(ang_splits) == self.num_lat_ang

        self.len_ubs, self.len_lbs, self.len_splits = len_ubs, len_lbs, len_splits
        self.ang_ubs, self.ang_lbs, self.ang_splits = ang_ubs, ang_lbs, ang_splits

        self.len_units = (self.len_ubs - self.len_lbs) / self.len_splits  # element-wise
        self.ang_units = (self.ang_ubs - self.ang_lbs) / self.ang_splits

        self.lat_len_prec = sum(self.len_splits)
        self.lat_ang_prec = sum(self.ang_splits)

        return

    def encode_latparam(self, lat_param, ub, lb, unit, split):
        lat_param_vec = np.zeros(shape=(split,))
        encoded_value = lat_param - lb
        if encoded_value < 0:
            log_printer.info(f'Lattice param {lat_param} smaller than {lb}')
            encoded_value = 0
        lat_bit = int(np.floor(encoded_value / unit))
        if lat_bit > split - 1:
            log_printer.info(f'Lattice param {lat_param} larger than {ub}')
            lat_bit = split - 1
        lat_param_vec[lat_bit] = 1
        return lat_param_vec
    
    def decode_latparam(self, lat_param_vec, lb, unit, decoding_fn):
        lat_bit = decoding_fn(lat_param_vec)
        lat_bit = lat_bit[0]  # note: choose the first ones
        lat_param = float(lat_bit * unit) + lb
        return lat_param


##### Encoder #####
        

class LatParamEncoder:
    def __init__(self, latparam_representor, dim, len_ubs, len_lbs, len_splits, ang_ubs, ang_lbs, ang_splits):
        self.lat_repr = latparam_representor(dim=dim, len_ubs=len_ubs, len_lbs=len_lbs, len_splits=len_splits, 
                                             ang_ubs=ang_ubs, ang_lbs=ang_lbs, ang_splits=ang_splits)
        self.dim = self.lat_repr.dim
        self.len_ubs, self.len_lbs, self.len_splits = self.lat_repr.len_ubs, self.lat_repr.len_lbs, self.lat_repr.len_splits
        self.ang_ubs, self.ang_lbs, self.ang_splits = self.lat_repr.ang_ubs, self.lat_repr.ang_lbs, self.lat_repr.ang_splits
        self.len_units, self.ang_units = self.lat_repr.len_units, self.lat_repr.ang_units
        self.lat_len_prec, self.lat_ang_prec = self.lat_repr.lat_len_prec, self.lat_repr.lat_ang_prec
        return
    
    @property
    def num_var(self):
        return self.lat_repr.num_var
    
    def encode_lat_len(self, pmg_struct):
        lat_param_vec = []
        for i, lat_len in enumerate(pmg_struct.lattice.lengths):  # a, b, c
            bit_vec = self.lat_repr.encode_latparam(lat_len, self.len_ubs[i], self.len_lbs[i], self.len_units[i], self.len_splits[i])
            lat_param_vec.append(bit_vec)
        return np.concatenate(lat_param_vec)
    
    def encode_lat_ang(self, pmg_struct):
        lat_param_vec = []
        if self.dim == 3:
            for i, lat_ang in enumerate(pmg_struct.lattice.angles):  # alpha, beta, gamma
                bit_vec = self.lat_repr.encode_latparam(lat_ang, self.ang_ubs[i], self.ang_lbs[i], self.ang_units[i], self.ang_splits[i])
                lat_param_vec.append(bit_vec)
        elif self.dim == 2:
            lat_angs = np.array(pmg_struct.lattice.angles)
            lat_ang_not_90 = lat_angs[np.round(lat_angs, 0) != 90]
            lat_ang = 90 if len(lat_ang_not_90) == 0 else lat_ang_not_90[0]  # not triggered since only dealing with perpendicular ones
            bit_vec = self.lat_repr.encode_latparam(lat_ang, self.ang_ubs[0], self.ang_lbs[0], self.ang_units[0], self.ang_splits[0])
            lat_param_vec.append(bit_vec)
        else:
            raise NotImplementedError
        return np.concatenate(lat_param_vec)
    
    def encode(self, pmg_struct):
        lat_len_vec = self.encode_lat_len(pmg_struct)
        lat_ang_vec = self.encode_lat_ang(pmg_struct)
        return [lat_len_vec, lat_ang_vec]


##### Decoder #####


class LatParamDecoder:
    def __init__(self, latparam_representor, dim, len_ubs, len_lbs, len_splits, ang_ubs, ang_lbs, ang_splits, decoding_fn, default_lat):
        self.lat_repr = latparam_representor(dim=dim, len_ubs=len_ubs, len_lbs=len_lbs, len_splits=len_splits, 
                                             ang_ubs=ang_ubs, ang_lbs=ang_lbs, ang_splits=ang_splits)
        self.dim = self.lat_repr.dim
        self.len_ubs, self.len_lbs, self.len_splits = self.lat_repr.len_ubs, self.lat_repr.len_lbs, self.lat_repr.len_splits
        self.ang_ubs, self.ang_lbs, self.ang_splits = self.lat_repr.ang_ubs, self.lat_repr.ang_lbs, self.lat_repr.ang_splits
        self.len_units, self.ang_units = self.lat_repr.len_units, self.lat_repr.ang_units
        self.lat_len_prec, self.lat_ang_prec = self.lat_repr.lat_len_prec, self.lat_repr.lat_ang_prec
        
        self.decoding_fn = decoding_fn
        self.default_len, self.default_ang = default_lat
        return
        
    def decode_lat_len(self, lat_len_vec):
        len_splits_cum = np.concatenate([[0], np.cumsum(self.len_splits)])
        lat_len_list = [lat_len_vec[len_splits_cum[i]: len_splits_cum[i + 1]] for i in range(len(self.len_splits))]
        lat = []
        for i, lat_len in enumerate(lat_len_list):
            try:
                lat_len = self.lat_repr.decode_latparam(lat_len, self.len_lbs[i], self.len_units[i], self.decoding_fn)
            except IndexError:
                print('Use default lattice length')
                lat_len = self.default_len
            lat.append(lat_len)
        return lat
    
    def decode_lat_ang(self, lat_ang_vec):
        ang_splits_cum = np.concatenate([[0], np.cumsum(self.ang_splits)])
        lat_ang_list = [lat_ang_vec[ang_splits_cum[i]: ang_splits_cum[i + 1]] for i in range(len(self.ang_splits))]
        lat = []
        for i, lat_ang in enumerate(lat_ang_list):
            try:
                lat_ang = self.lat_repr.decode_latparam(lat_ang, self.ang_lbs[i], self.ang_units[i], self.decoding_fn)
            except IndexError:
                print('Use default lattice angle')
                lat_ang = self.default_ang
            lat.append(lat_ang)
        return lat
    
    def decode(self, lat_vec):
        lat_len_vec = lat_vec[0: self.lat_len_prec]
        lat_ang_vec = lat_vec[self.lat_len_prec:]
        lat_len = self.decode_lat_len(lat_len_vec)  # list
        lat_ang = self.decode_lat_ang(lat_ang_vec)  # list
        if self.dim == 2:
            lat_ang = [90, 90] + lat_ang
        lat_mat = para2matrix(cell_para=lat_len + lat_ang, radians=False, format='lower')
        default_lat = 8
        if np.isnan(np.sum(lat_mat)):  # para2matrix can lead to nan
            lat_mat[np.isnan(lat_mat)] = default_lat
        lat_mat = np.where(np.abs(lat_mat) > 1e-4, lat_mat, 0)
        return lat_mat


# decode lat params to make them consistent with a crystal system
class LatParamCrysSysDecoder:
    def __init__(self, latparam_representor, dim, len_ubs, len_lbs, len_splits, ang_ubs, ang_lbs, ang_splits, decoding_fn, default_lat):
        self.lat_repr = latparam_representor(dim=dim, len_ubs=len_ubs, len_lbs=len_lbs, len_splits=len_splits, 
                                             ang_ubs=ang_ubs, ang_lbs=ang_lbs, ang_splits=ang_splits)
        self.dim = self.lat_repr.dim
        self.len_ubs, self.len_lbs, self.len_splits = self.lat_repr.len_ubs, self.lat_repr.len_lbs, self.lat_repr.len_splits
        self.ang_ubs, self.ang_lbs, self.ang_splits = self.lat_repr.ang_ubs, self.lat_repr.ang_lbs, self.lat_repr.ang_splits
        self.len_units, self.ang_units = self.lat_repr.len_units, self.lat_repr.ang_units
        self.lat_len_prec, self.lat_ang_prec = self.lat_repr.lat_len_prec, self.lat_repr.lat_ang_prec
        
        self.decoding_fn = decoding_fn
        self.default_len, self.default_ang = default_lat
        return
    
    def decode_lat_len(self, lat_len_vec, crys_sys):
        len_splits_cum = np.concatenate([[0], np.cumsum(self.len_splits)])
        lat_len_list = [lat_len_vec[len_splits_cum[i]: len_splits_cum[i + 1]] for i in range(len(self.len_splits))]
        lat = []
        for i, lat_len in enumerate(lat_len_list):
            try:
                lat_len = self.lat_repr.decode_latparam(lat_len, self.len_lbs[i], self.len_units[i], self.decoding_fn)
            except IndexError:
                print('Use default lattice length')
                lat_len = self.default_len
            if i == 0:
                lat_len_a = lat_len
            if i == 1:  # b
                if crystal_system_dict(dim=self.dim)[crys_sys]['b'] == 'a':
                    lat_len = lat_len_a
            if i == 2:  # c
                if crystal_system_dict(dim=self.dim)[crys_sys]['c'] == 'a':
                    lat_len = lat_len_a
            lat.append(lat_len)
        return lat

    def decode_lat_ang(self, lat_ang_vec, crys_sys):
        ang_splits_cum = np.concatenate([[0], np.cumsum(self.ang_splits)])
        lat_ang_list = [lat_ang_vec[ang_splits_cum[i]: ang_splits_cum[i + 1]] for i in range(len(self.ang_splits))]
        lat = []
        if self.dim == 3:
            angle_name = ['alpha', 'beta', 'gamma']
        elif self.dim == 2:
            angle_name = ['gamma']
        else:
            raise NotImplementedError
        for i, lat_ang in enumerate(lat_ang_list):
            preassigned_ang = crystal_system_dict(dim=self.dim)[crys_sys][angle_name[i]]
            if preassigned_ang is not None:
                lat_ang = preassigned_ang
            else:
                try:
                    lat_ang = self.lat_repr.decode_latparam(lat_ang, self.ang_lbs[i], self.ang_units[i], self.decoding_fn)
                except IndexError:
                    print('Use default lattice angle')
                    lat_ang = self.default_ang
            lat.append(lat_ang)
        return lat
    
    def decode(self, lat_vec, crys_sys):
        lat_len_vec = lat_vec[0: self.lat_len_prec]
        lat_ang_vec = lat_vec[self.lat_len_prec:]
        lat_len = self.decode_lat_len(lat_len_vec, crys_sys)  # list
        lat_ang = self.decode_lat_ang(lat_ang_vec, crys_sys)  # list
        if self.dim == 2:
            lat_ang = [90, 90] + lat_ang
        lat_mat = para2matrix(cell_para=lat_len + lat_ang, radians=False, format='lower')
        default_lat = 8
        if np.isnan(np.sum(lat_mat)):  # para2matrix can lead to nan
            lat_mat[np.isnan(lat_mat)] = default_lat
        lat_mat = np.where(np.abs(lat_mat) > 1e-4, lat_mat, 0)
        return lat_mat


from crysim.src.registry import OptimizerName
from crysim.src.crysvec.utils import crys_embbeder
from functools import partial
oh_name = OptimizerName.onehot()[0]
crys_embbeder['lat'][f'{oh_name}-basic'] = {
    'enc': partial(LatParamEncoder, latparam_representor=LatParamDiscretizer), 
    'dec': partial(LatParamDecoder, latparam_representor=LatParamDiscretizer)
}
crys_embbeder['lat'][f'{oh_name}-cs'] = {
    'enc': partial(LatParamEncoder, latparam_representor=LatParamDiscretizer), 
    'dec': partial(LatParamCrysSysDecoder, latparam_representor=LatParamDiscretizer)
}

crys_embbeder['lat'][f'lp'] = {
    'enc': partial(LatParamEncoder, latparam_representor=LatParam), 
    'dec': partial(LatParamDecoder, latparam_representor=LatParam)
}
crys_embbeder['lat'][f'lp-cs'] = {
    'enc': partial(LatParamEncoder, latparam_representor=LatParam), 
    'dec': partial(LatParamCrysSysDecoder, latparam_representor=LatParam)
}
