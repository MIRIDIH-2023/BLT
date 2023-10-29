import splitfolders
import tqdm
input_path = "/home/work/increased_en_data/BLT/all_data"
output_path = "/home/work/increased_en_data/BLT/data"

# tqdm(splitfolders.ratio(input_path, output=output_path, seed=77, ratio=(0.8, 0.1, 0.1)))


# count dir files
import os

print("test: ", len(os.listdir("/home/work/increased_en_data/BLT/data/test/json_data")))
print("val: ",len(os.listdir("/home/work/increased_en_data/BLT/data/val/json_data")))
print("train: ",len(os.listdir("/home/work/increased_en_data/BLT/data/train/json_data")))