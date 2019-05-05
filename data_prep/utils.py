import os
import json
import pickle

def _subarea_to_number(s):
    s = s.replace("\n", "")
    s_list = s.split("#")
    s_list[1] = int(s_list[1])
    return s_list

def prepare_sub_area_dummy_dict():
    CUR_SCRIPT_PATH = os.path.abspath(os.path.normpath(os.path.join(__file__, '../')))
    TXT_DICT_PATH = os.path.abspath(os.path.normpath(
        os.path.join(CUR_SCRIPT_PATH, '../configs/subarea_dict.txt')))
    with open(TXT_DICT_PATH, 'r') as f:
        sub_areas = f.readlines()
    sub_area_list = list(map(_subarea_to_number, sub_areas))
    return dict(sub_area_list)

def prepare_dict_build_year_from_sub_area():
    CUR_SCRIPT_PATH = os.path.abspath(os.path.normpath(os.path.join(__file__, '../')))
    build_year_DICT_PATH = os.path.abspath(os.path.normpath(
        os.path.join(CUR_SCRIPT_PATH, '../configs/build_year_from_sub_area_dict.json')))
    with open(build_year_DICT_PATH) as json_file:
        build_year_from_sub_area_dict = json.load(json_file)
    return build_year_from_sub_area_dict

def prepare_dict_max_floor_from_build_year():
    CUR_SCRIPT_PATH = os.path.abspath(os.path.normpath(os.path.join(__file__, '../')))
    max_floor_DICT_PATH = os.path.abspath(os.path.normpath(
        os.path.join(CUR_SCRIPT_PATH, '../configs/max_floor_from_build_year_dict.json')))
    with open(max_floor_DICT_PATH) as json_file:
        max_floor_from_build_year_dict = json.load(json_file)
    return {int(k): v for k, v in max_floor_from_build_year_dict.items()}

def prepare_linear_model_for_life_sq():
    CUR_SCRIPT_PATH = os.path.abspath(os.path.normpath(os.path.join(__file__, '../')))
    life_sq_MODEL_PATH = os.path.abspath(os.path.normpath(
        os.path.join(CUR_SCRIPT_PATH, '../trained_models/data_prep_models/reg_life_sq.dat')))
    model_life_sq = pickle.load(open(life_sq_MODEL_PATH, "rb"))
    return model_life_sq

def prepare_linear_model_for_kitch_sq():
    CUR_SCRIPT_PATH = os.path.abspath(os.path.normpath(os.path.join(__file__, '../')))
    kitch_sq_MODEL_PATH = os.path.abspath(os.path.normpath(
        os.path.join(CUR_SCRIPT_PATH, '../trained_models/data_prep_models/reg_kitch_sq.dat')))
    model_kitch_sq = pickle.load(open(kitch_sq_MODEL_PATH, "rb"))
    return model_kitch_sq

if __name__=="__main__":

    prepare_sub_area_dummy_dict()

    print()
