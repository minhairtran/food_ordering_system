# This file is for getting just 3 audios from 13 audios generated. 
# The reason is that model can overfit the voice from which these 
# files are generated 

import sys
sys.path.append("../")

import os
from shutil import copyfile
import random

OLD_DATASET_PATH = ["../../data/food_dataset/ca_kho_1", "../../data/food_dataset/ca_xot_1", "../../data/food_dataset/khoai_tay_chien_1", "../../data/food_dataset/com_heo_xi_muoi_1", "../../data/food_dataset/com_nieu_1", \
                    "../../data/food_dataset/com_tam_1", "../../data/food_dataset/com_thap_cam_1", "../../data/food_dataset/khong_biet_1", "../../data/food_dataset/rau_cai_luoc_1", \
                        "../../data/food_dataset/rau_cai_xao_1",  "../../data/food_dataset/salad_tron_1", "../../data/food_dataset/tra_hoa_cuc_1", "../../data/food_dataset/tra_sam_dua_1", \
                            "../../data/food_dataset/trung_chien_1"]
NEW_DATASET_PATH = {
    "../../data/food_dataset/ca_kho_1": "../../data/food_dataset/ca_kho", 
    "../../data/food_dataset/ca_xot_1": "../../data/food_dataset/ca_xot", 
    "../../data/food_dataset/khoai_tay_chien_1": "../../data/food_dataset/khoai_tay_chien", 
    "../../data/food_dataset/com_heo_xi_muoi_1": "../../data/food_dataset/com_heo_xi_muoi", 
    "../../data/food_dataset/com_nieu_1": "../../data/food_dataset/com_nieu",
    "../../data/food_dataset/com_tam_1": "../../data/food_dataset/com_tam", 
    "../../data/food_dataset/com_thap_cam_1": "../../data/food_dataset/com_thap_cam", 
    "../../data/food_dataset/khong_biet_1": "../../data/food_dataset/khong_biet", 
    "../../data/food_dataset/rau_cai_luoc_1": "../../data/food_dataset/rau_cai_luoc",
    "../../data/food_dataset/rau_cai_xao_1": "../../data/food_dataset/rau_cai_xao",  
    "../../data/food_dataset/salad_tron_1": "../../data/food_dataset/salad_tron", 
    "../../data/food_dataset/tra_hoa_cuc_1": "../../data/food_dataset/tra_hoa_cuc", 
    "../../data/food_dataset/tra_sam_dua_1": "../../data/food_dataset/tra_sam_dua",
    "../../data/food_dataset/trung_chien_1": "../../data/food_dataset/trung_chien"
    }

# OLD_DATASET_PATH = ["../../data/confirming_dataset/khong_biet_1/"]
# NEW_DATASET_PATH = { 
#     "../../data/confirming_dataset/khong_biet_1/": "../../data/confirming_dataset/khong_biet/",
#     }


def move_dataset(old_dataset_path, new_dataset_path):
    for old_dataset in old_dataset_path:
        # loop through all sub-dirs
        for i, (old_dirpath, old_dirnames, old_filenames) in enumerate(os.walk(old_dataset)):

            label = old_dirpath.split("/")[-1]

            print("\nProcessing: '{}'".format(label))

            file_speed = {"speed": 0, "files": []}

            for old_file in old_filenames:
                old_path = os.path.join(old_dirpath, old_file)

                #slow speed
                if file_speed["speed"] < 6:
                    file_speed["files"].append(old_path)
                if file_speed["speed"] == 5:
                    chosen_file = random.choice(file_speed["files"])
                    if not os.path.exists(new_dataset_path[old_dataset]):
                        os.makedirs(new_dataset_path[old_dataset])
                    new_path = os.path.join(new_dataset_path[old_dataset], chosen_file.split("/")[-1])
                    copyfile(old_path, new_path)
                    file_speed["files"] = []
                
                #normal speed
                if file_speed["speed"] == 6:
                    new_path = os.path.join(new_dataset_path[old_dataset], old_file)
                    copyfile(old_path, new_path)
                
                #fast speed
                if file_speed["speed"] < 13 and file_speed["speed"] > 6:
                    file_speed["files"].append(old_path)
                if file_speed["speed"] == 12:
                    chosen_file = random.choice(file_speed["files"])
                    new_path = os.path.join(new_dataset_path[old_dataset], chosen_file.split("/")[-1])
                    copyfile(old_path, new_path)
                    file_speed["files"] = []

                if file_speed["speed"] < 12:
                    file_speed["speed"] += 1
                else:
                    file_speed["speed"] = 0

                #khong biet
                # if file_speed["speed"] < 13:
                #     file_speed["files"].append(old_path)
                # if file_speed["speed"] == 12:
                #     chosen_files = random.sample(file_speed["files"], k=3)
                #     for chosen_file in chosen_files:
                #         new_path = os.path.join(new_dataset_path[old_dataset], chosen_file.split("/")[-1])
                #         copyfile(old_path, new_path)
                #     file_speed["files"] = []
                # if file_speed["speed"] < 12:
                #     file_speed["speed"] += 1
                # else:
                #     file_speed["speed"] = 0
                

if __name__ == "__main__":
    move_dataset(OLD_DATASET_PATH, NEW_DATASET_PATH)