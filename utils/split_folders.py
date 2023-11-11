import splitfolders
from tqdm import tqdm
import shutil
import os
import pickle
import json
import shutil

input_path = "/home/work/increased_en_data/BLT/posData"
output_path = "/home/work/increased_en_data/BLT/data2"

# files = os.listdir(input_path)
# for file_ in tqdm(files) :
#     source_file = os.path.join(input_path, file_)
#     with open(source_file, 'rb') as f:
#         try:
#             obj = pickle.load(f)
#             json_data = json.loads(json.dumps(obj, default=str))
#         except:
#             raise AssertionError(f"Wrong file: {source_file}")
#     if(not json_data["no_RendorPos"]) :
#         shutil.copy(source_file, output_path)
#     else :
#         print(file_)

# tqdm(splitfolders.ratio(input_path, output=output_path, seed=77, ratio=(0.8, 0.1, 0.1)))


# count dir files
# import os

print("test: ", len(os.listdir("/home/work/increased_en_data/BLT/data2/test/json_data")))
print("val: ",len(os.listdir("/home/work/increased_en_data/BLT/data2/val/json_data")))
print("train: ",len(os.listdir("/home/work/increased_en_data/BLT/data2/train/json_data")))
# print("data len: ", len(os.listdir("/home/work/increased_en_data/BLT/posData/json_data")))