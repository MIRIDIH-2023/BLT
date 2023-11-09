import pickle
import json
import math
import os
from tqdm import tqdm

def load_text_infos(json_data, im_width) :
    text_area = 0
    text_coord_lr = []
    text_coord_tb = []
    text_info_pair = []
    text_size_db = set()
    for text_info in json_data["form"] :
        # rotate가 0이 아니면 텍스트 데이터로 사용하지 않음
        if(text_info["rotate"] != "0" or text_info["text"] == None or text_info["text"] == "" or text_info["text"] == " ") : continue
        left, top, right, bottom = text_info["box"]

        center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
        width_ = right - left
        height_ = bottom - top

        doc_width , doc_height = text_info["sheet_size"]
        RATIO = im_width / doc_width

        # text size와 line space 추출
        ts = min(text_info["font_size"])
        ls = min(text_info["linespace"])
        ratio_text_size = ts * RATIO
        line_space = ls * RATIO

        if ratio_text_size <= 0 : continue

        split_cnt1 = len(text_info["text"].split('\n'))
        split_cnt2 = len(text_info["text"].split('\\n'))
        split_cnt = split_cnt1 if split_cnt1 > split_cnt2 else split_cnt2
        if (ratio_text_size + line_space) <= 0 : line_space = 0
        line_nums = int(math.floor(float(height_ / (ratio_text_size + line_space))))
        line_nums = line_nums if line_nums > 0 else 1
        line_nums = split_cnt if split_cnt >= line_nums else line_nums

        text_size = int(math.floor(ts))
        if text_size <= 0 : continue

        # text_size와 line_nums, idx 쌍 만들기
        text_info_pair.append((text_size, line_nums, center_x, center_y, width_, height_))

        # text bbox 좌표 저장하기
        text_coord_lr.append([left, right])
        text_coord_tb.append([top, bottom])

        # text 영역 넓이 구하기
        text_area += width_ * height_

        # text size db 저장
        text_size_db.add(text_size)

    text_infos = {
        "text_area" : text_area,
        "text_coord_lr" : text_coord_lr,
        "text_coord_tb" : text_coord_tb,
        "text_info_pair" : text_info_pair,
        "text_size_db" : text_size_db
    }

    return text_infos

def load_categorized(path, label_names, label_to_id, idx, with_background_test=False) :
    data = []
    image_link=None

    file_name_lst = os.listdir(path)
    file_name_lst = [file_name_lst[idx]] if with_background_test else tqdm(file_name_lst) 
    for file_name in file_name_lst :
        # pickle file open
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as f:
            try:
                obj = pickle.load(f)
                json_data = json.loads(json.dumps(obj, default=str))
            except:
                raise AssertionError(f"Wrong file: {file_path}")

        im_width, im_height = json_data["thumbnail_size"]
        template = {
            "children" : [],
            "width" : im_width,
            "height" : im_height,
        }

        text_infos =  load_text_infos(json_data=json_data, im_width=im_width)

        text_area = text_infos["text_area"]
        text_coord_lr = text_infos["text_coord_lr"]
        text_coord_tb = text_infos["text_coord_tb"]
        text_info_pair = text_infos["text_info_pair"]
        text_size_db = text_infos["text_size_db"]

        # db 내림차순 정렬하기 
        text_size_db = list(text_size_db)
        text_size_db.sort(reverse=True)
        text_size_db = text_size_db if len(text_size_db) <= 4 else text_size_db[:4]

        # text data 저장
        for pair in text_info_pair :
            label = "text_"
            for idx, size in enumerate(text_size_db) :
                if pair[0] >= size :
                    label += f"{idx}_{pair[1]}" if pair[1] < 4 else f"{idx}_3_over"
                    break
                
                if idx == 3 and label == "text_" :
                    label += f"3_{pair[1]}" if pair[1] < 4 else "3_3_over"
                    
            template["children"].append({
                "category_id" : label_to_id[label],
                "center" : [pair[2], pair[3]],
                "width" : pair[4],
                "height" : pair[5]
            })

        # load tags
        for info in json_data["tags_info"] :
            if info["tag"] not in label_names or info["rotate"] != "0": continue

            left, top, right, bottom = info["box"]
            center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
            width_ = right - left
            height_ = bottom - top

            if (width_ * height_) < (text_area / 2) : continue

            lr_dup = False
            tb_dup = False
            is_back_lr = True
            is_back_tb = True
            for lr, tb in zip(text_coord_lr, text_coord_tb) :
                if (lr[0] < left and left < lr[1]) or (lr[0] < right and right < lr[1]) :
                    lr_dup = True
                if (tb[0] < top and top < tb[1]) or (tb[0] < bottom and bottom < tb[1]) :
                    tb_dup = True
                if (lr[0] < left or right < lr[1]) :
                    is_back_lr = False
                if (tb[0] < top or bottom < tb[1]) :
                    is_back_tb = False

            if lr_dup or tb_dup : continue

            if info["tag"] == "Chart" or info["tag"] == "GRID":
                label = info["tag"]
            else :
                if is_back_lr and is_back_tb :
                    label = "background"
                else :
                    label = "image"

            template["children"].append({
                "category_id" : label_to_id[label],
                "center" : [center_x, center_y],
                "width" : width_,
                "height" : height_
            })

        if with_background_test : image_link = json_data["thumbnail_url"]

        data.append(template)
    
    return data, image_link

def miri_load(path, label_names, label_to_id) :
    data = []
    file_name_lst = tqdm(os.listdir(path))
    for file_name in file_name_lst :
        # pickle file open
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as f:
            try:
                obj = pickle.load(f)
                json_data = json.loads(json.dumps(obj, default=str))
            except:
                raise AssertionError(f"Wrong file: {file_}")

        template = {
            "children" : [],
            "width" : json_data["thumbnail_size"][0],
            "height" : json_data["thumbnail_size"][1],
        }

        # load tags
        for info in json_data["tags_info"] :
            if info["tag"] not in label_names: continue
            
            left, top, right, bottom = info["box"]
            center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
            width_ = right - left
            height_ = bottom - top
            template["children"].append({
                "category_id" : label_to_id[info["tag"]],
                "center" : [center_x, center_y],
                "width" : width_,
                "height" : height_
            })

        # load texts
        for text_info in json_data["form"] :
            left, top, right, bottom = text_info["raw_bbox"]
            center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
            width_ = right - left
            height_ = bottom - top
            template["children"].append({
                "category_id" : label_to_id["TEXT"],
                "center" : [center_x, center_y],
                "width" : width_,
                "height" : height_
            })

        data.append(template)

    return data