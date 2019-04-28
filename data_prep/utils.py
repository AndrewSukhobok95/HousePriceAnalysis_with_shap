import os

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


if __name__=="__main__":

    prepare_sub_area_dummy_dict()

    print()
