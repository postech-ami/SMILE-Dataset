import os
import pdb

import openai
import argparse
import random
import numpy as np
import torch
import json
import os
from tqdm import tqdm
import sys
sys.path.append("./caption_evaluation")
from eval_metrics import evaluate_metrics_total

def fewshot_metadata(engine, shot1, shot2, shot3, query):
    prompt = f"{shot1}###{shot2}###{shot3}###{query}"
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.5,
        max_tokens=200,
    )
    answer = response['choices'][0]['text']

    return answer

def zeroshot_metadata(engine, query):
    response = openai.Completion.create(
        engine=engine,
        prompt=query,
        temperature=0.5,
        max_tokens=200,
    )
    answer = response['choices'][0]['text']

    return answer

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def zeroshot(args):
    with open(args.val_data, "r") as f:
        validation_data = json.load(f)
    pred_caption = []
    gt_caption = []

    for i in tqdm(range(len(validation_data))):
        temp = 1
        inputs = validation_data[i]['conversations'][0]['value'] + "### Assistant:"

        while temp:
            #in case if there is network error while connecting to openai api
            try:
                answer = zeroshot_metadata(args.engine, inputs)
                temp = None
            except:
                temp = 1
        answer=answer.replace("\n", "")
        #define dictionary
        pred_dict={"file_name":i, "caption_predicted":answer}
        gt_dict = {"file_name":i, "caption_reference_01":validation_data[i]['conversations'][1]['value']}

        pred_caption.append(pred_dict)
        gt_caption.append(gt_dict)
    evaluate_metrics_total(pred_caption, gt_caption, 1)

def fewshot(args):
    with open(args.val_data, "r") as f:
        validation_data = json.load(f)
    with open(args.train_data, "r") as f:
        train_data = json.load(f)
    total = len(train_data)
    pred_caption = []
    gt_caption = []

    for i in tqdm(range(len(validation_data))):
        rand_list = random.sample(range(0, total), k=3)
        shot1 = train_data[rand_list[0]]["conversations"][0]["value"] + "### Assistant:" +train_data[rand_list[0]]["conversations"][1]["value"]
        shot2 = train_data[rand_list[1]]["conversations"][0]["value"] + "### Assistant:" + \
                train_data[rand_list[1]]["conversations"][1]["value"]
        shot3 = train_data[rand_list[2]]["conversations"][0]["value"] + "### Assistant:" + \
                train_data[rand_list[2]]["conversations"][1]["value"]

        temp = 1
        inputs = validation_data[i]['conversations'][0]['value'] + "### Assistant:"
        while temp:
            try:
                answer = fewshot_metadata(args.engine, shot1, shot2, shot3, inputs)
                temp = None
            except:
                rand_list = random.sample(range(0, total), k=3)
                shot1 = train_data[rand_list[0]]["conversations"][0]["value"] + "### Assistant:" + \
                        train_data[rand_list[0]]["conversations"][1]["value"]
                shot2 = train_data[rand_list[1]]["conversations"][0]["value"] + "### Assistant:" + \
                        train_data[rand_list[1]]["conversations"][1]["value"]
                shot3 = train_data[rand_list[2]]["conversations"][0]["value"] + "### Assistant:" + \
                        train_data[rand_list[2]]["conversations"][1]["value"]
                temp = 1

        answer = answer.replace("\n", "")

        pred_dict={"file_name":i, "caption_predicted":answer}
        gt_dict = {"file_name":i, "caption_reference_01":validation_data[i]['conversations'][1]['value']}

        pred_caption.append(pred_dict)
        gt_caption.append(gt_dict)

    evaluate_metrics_total(pred_caption, gt_caption, 1)



def main(args):
    if args.shot=="zeroshot":
        zeroshot(args)
    elif args.shot=="fewshot":
        fewshot(args)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-openai_key',default=None, help="Enter your openai api key")
    parser.add_argument('-engine', default="text-davinci-002", help="Enter your openai api key")
    parser.add_argument('-shot', default="zeroshot", help="Choose between zeroshot or fewshot")
    parser.add_argument('-val_data', default="SMILE_v1_evaluation/sitcom_reasoning_val.json", help="Enter the validation data path")
    parser.add_argument('-train_data', default="SMILE_v1_evaluation/sitcom_reasoning_train.json",help="Enter the training data path")
    parser.add_argument('-random_seed', default=1234,type=int, help="random seed")

    args = parser.parse_args()
    openai.api_key = args.openai_key
    seed_everything(args.random_seed)
    main(args)


