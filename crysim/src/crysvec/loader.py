import numpy as np
from functools import partial

from crysim.src.crysvec.emb import emb_type_dict, MatProcessor
from crysim.src.crysvec.emb_oh_mix import MatCSSPGMixProcessor
from crysim.src.crysvec.utils import find_bit_from_vec, find_bit_from_mat, top_k_from_mat
from crysim.src.struct_utils import unique_elements


def standardize_splits(splits, len_splits, ang_splits, dim, elements, emb_type, use_mixing):
    coords_type, sym_type, enc_obj = emb_type.split('+')
    if not isinstance(splits, list):  # ex., 10
        if (enc_obj == 'used_atoms' and coords_type == 'latdis') or use_mixing:
            splits = [splits] * 3
        else:
            splits = [[splits] * 3 for _ in elements]
    else:  # ex., [[5, 5, 5], [10, 10, 10]]
        if not isinstance(splits[0], list):  # ex., [10, 10, 10]
            if not ((enc_obj == 'used_atoms' and coords_type == 'latdis') or use_mixing):
                splits = [splits for _ in elements]
    if not isinstance(len_splits, list):
        if len_splits is not None:
            len_splits = [len_splits] * 3
    if not isinstance(ang_splits, list):
        if ang_splits is not None:
            if dim == 3:
                ang_splits = [ang_splits] * 3
    return splits, len_splits, ang_splits


class EmbedderProxy:
    def __init__(self, species_list, emb_type, is_var_bit, dim, splits, len_splits, ang_splits, decode_num, wpc_prec, include_spg_list,
                 use_mixing=False):
        assert emb_type in emb_type_dict, f"Current supported emb_type: {list(emb_type_dict.keys())}, input: {emb_type}"
        self.species_list = species_list
        self.emb_type = emb_type
        self.is_var_bit = is_var_bit
        self.dim = dim
        self.wpc_prec = wpc_prec
        self.decode_num = decode_num
        self.use_mixing = use_mixing
        self.include_spg_list = include_spg_list
        
        elements, _ = unique_elements(species_list)
        splits, len_splits, ang_splits = standardize_splits(splits, len_splits, ang_splits, dim, elements, emb_type, use_mixing)
        self.splits = splits
        self.len_splits, self.ang_splits = len_splits, ang_splits
        
        splits = np.array(splits).reshape(1, -1) if np.array(splits).ndim == 1 else np.array(splits)
        max_splits = np.max(splits, axis=1)
        max_splits = [str(i) for i in max_splits]
        var_type = 'bit' if self.is_var_bit else 'cont'
        self.model_name = f"{self.emb_type}-{var_type}{''.join(max_splits)}-{decode_num}"
        self.real_model_name = f"{self.emb_type}-{var_type}Precs{''.join(max_splits)}-DecNum{decode_num}"
        return
    
    def unwrap(self, len_lbs, len_ubs, ang_lbs, ang_ubs):
        decoding_fn_vec = find_bit_from_vec
        decoding_fn_mat = find_bit_from_mat if self.is_var_bit else partial(top_k_from_mat, k=len(self.species_list))
        
        if not self.use_mixing:
            # sig = inspect.signature(embedder.__init__).parameters
            # kwargs = {key: value for key, value in kwargs.items() if key in sig}
            # if 'decode_num' in kwargs:
            #     kwargs['decode_num'] = self.decode_num
            embedder = MatProcessor(emb_type=self.emb_type, dim=self.dim, species_list=self.species_list, splits=self.splits, 
                                    len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                                    len_splits=self.len_splits, ang_splits=self.ang_splits,
                                    decode_num=self.decode_num, decoding_fn=(decoding_fn_vec, decoding_fn_mat), 
                                    wpc_prec=self.wpc_prec, include_spg_list=self.include_spg_list)
        else:
            embedder = MatCSSPGMixProcessor(dim=self.dim, species_list=self.species_list, splits=self.splits, 
                                            len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                                            len_splits=self.len_splits, ang_splits=self.ang_splits,
                                            decode_num=self.decode_num, decoding_fn=(decoding_fn_vec, decoding_fn_mat), 
                                            wpc_prec=self.wpc_prec, include_spg_list=self.include_spg_list)
        
        oh_settings = {
            'num_var': embedder.num_var, 'splits': embedder.splits, 
            'pos_prec': embedder.pos_prec, 'coords_prec': embedder.coords_prec, 'lat_prec': embedder.lat_prec,
            'len_splits': embedder.len_splits, 'len_ubs': embedder.len_ubs, 'len_lbs': embedder.len_lbs,
            'ang_splits': embedder.ang_splits, 'ang_ubs': embedder.ang_ubs, 'ang_lbs': embedder.ang_lbs,
        }
        oh_vars = [embedder.len_splits, embedder.ang_splits]
        if hasattr(embedder, 'cs_prec'):
            oh_settings['cs_prec'] = embedder.cs_prec
            oh_vars.append([embedder.cs_prec])
        if hasattr(embedder, 'spg_prec'):
            oh_settings['spg_prec'] = embedder.spg_prec
            oh_vars.append([embedder.spg_prec])
        if hasattr(embedder, 'wpc_prec'):
            oh_settings['wpc_prec'] = embedder.wpc_prec
            oh_vars.append([embedder.wpc_prec])
        oh_vars.append(embedder.coords_prec)
        oh_vars = np.concatenate(oh_vars)
        oh_settings['oh_vars'] = oh_vars

        return embedder, oh_settings
