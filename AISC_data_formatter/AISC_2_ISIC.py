#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:23:48 2020

@author: mohan
"""

diag_dict = {}
isic_label_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
isic_idxs = dict(zip(isic_label_names, range(len(isic_label_names))))

diags = {}
diags['NV'] = ('compound_nevus', 'benign_nevus', 'intraepidermal_nevus', 'dermal_nevus',
               'blue_nevus', 'congenital_nevus', 'halo_nevi', 'spitz_nevus',
               'dysplastic_nevus', 'nevus_ito', 'nevus_spilus', 'nevus_recurrens')

diags['MEL'] = ('superficially_spreading_melanoma', 'melanoma_in_situ', 'lentigo_maligna',
            'lentigo_maligna_melanoma', 'nodular_melanoma', 'dermal_melanoma')

diags['BCC'] = ('basal_cell_carcinoma',)
diags['SCC'] = ('squamous_cell_carcinoma',)
diags['DF'] = ('dermatofibroma',)
diags['AK'] = ('bowens_disease', 'actinic_keratosis')

diags['VASC'] = ('hemangioma', 'hemorrhage', 'pyogenic_granuloma', 'venous_lake',
        'other_vascular_lesion', 'other_vascular_tumors', 'purpura' 'telangiectasia',
        'spider_angioma', 'traumatic_hemorrhage')

#SK
diags['BKL'] = ('seborrheic_keratosis', 'lentigo_solaris', 'lichen_planus_like_keratosis')

#Inflammation
diags[-1] = ('hyperpigmentation', 'acute_chronic_inflammation', 'lichenoid_keratosis',
          'post_inflammatory_pigmentation', 'pigmentation', 'dermatitis', 
          'pigmentation_without_known_cause', 'lichen_planus', 'pigmentation_unknown_etiology',
          'other_inflammatory_reaction', 'lichen_simplex_chronicus')
#Other
diags[-2] = ('labial_melanotic_macula', 'lentigo_simplex', 'melanonychia_striata')

#Forgotten
diags[-3] = ('squamous_cell_papilloma', 'ephelides', 'dilated_follicle', 'hyperkeratosis', 
             'kyrles_disease', 'pseudolymfoma', 'lymphangioma', 'acanthosis', 'ateroma', 
             'inverted_follicular_keratosis', 'pseudolymfoma', 'sebaceous_hyperplasia', 
             'epithelial_inclusion_cyst', 'cicatrix', 'ulcus',  'absces', 'telangiectasia',
             'atypical_fibroxanthoma', 'pilomatricoma', 'other_benign_lesion', 'supernumerary_nipple',
             'folliculitis', 'large_cell_acanthoma', 'xanthelasma', 'sebaceous_adenoma', 'neurofibroma', 
             'grovers_disease', 'purpura')
new_diganoses = ('keratoacanthoma', 'senile_purpura')

diags[-3] += new_diganoses

for diag_key, diag_list in diags.items():
    if not isinstance(diag_key, int):
        diag_key = isic_label_names.index(diag_key)
    else:
        assert(diag_key<0)
    for diag in diag_list:
        diag_dict[diag] = diag_key


# number_list = [diag_dict[i] for i in diag_dict.keys()]
# breakpoint()
# data['images'][1]['lesion'].keys()