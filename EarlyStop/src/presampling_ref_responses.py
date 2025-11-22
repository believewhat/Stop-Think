import os, sys, json, jsonlines
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import traceback
import requests
from copy import deepcopy
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10) # 0 for all data
    parser.add_argument("--K", type=int, default=16) 
    parser.add_argument("--dataset_path", type=str, default='./data/test/deepscaler.json')  # data path
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-1.5B")  
    parser.add_argument("--max_tokens", type=int, default=16384)   
    parser.add_argument("--proc_num", type=int, default=512)  
    parser.add_argument("--nothinking", action='store_true', default=False)   
    
    return parser.parse_args()

args = parse_args()
K = args.K
model = args.model_name
ipt_path = args.dataset_path
dataset = args.dataset_path.split('/')[-1].split('.json')[0].strip()
num_samples = args.num_samples
max_tokens = args.max_tokens
nothinking = args.nothinking
        
print("INFERENCE:",K,model,dataset,"num:",num_samples)
suffix = '_nothinking' if nothinking else ''
fout_path = f"./data/train/ref_presampling/{model}_{dataset}_n{num_samples}_K{K}_len{max_tokens}{suffix}.jsonl"
os.makedirs("./data/train/ref_presampling", exist_ok=True)
print("Save to:", fout_path)
s = set()
if os.path.exists(fout_path):
    with jsonlines.open(fout_path, 'r') as f:
        for js in tqdm(f):
            s.add(js['_id'])

data = json.load(open(ipt_path,"r"))
random.seed(42)
random.shuffle(data)
if num_samples > 0:
    data = data[0:num_samples]
print("Used num problem:", len(data))

need_list = []
for i, js in enumerate(data):
    for j in range(K):
        new_js = deepcopy(js)
        new_js['_id'] = f'{i}_{j}'
        if new_js['_id'] not in s:
            need_list.append(new_js)

# need_list = need_list[:1]
print('Total inference num:', len(need_list))

if 'DeepSeek' in model:
    if nothinking:
        prompt_template = '<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n</think>'
    else:
        prompt_template = '<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n'
    # sampling_params['stop'] = ["<｜User｜>", "<｜end▁of▁sentence｜>"]

def chat(js):
    try:
        prompt = js['problem']
        request_data = {
            'model': model,
            "prompt": prompt_template.format(question=prompt),
            'max_tokens': max_tokens - (nothinking == 1),
            'temperature': 0.6,
            'top_p': 0.95,
            "stream": False,
        }
        response = requests.post(
            'http://127.0.0.1:8000/v1/completions',
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=1200,
            stream=False,
        )
        output = response.json()
        if output:
            # js['response'], js['usage'] = output, usage
            js['response'] = output
            with open(fout_path, "a") as fout:
                fout.write(json.dumps(js, ensure_ascii=False)+'\n')
                fout.flush()
            return 1   
    except:
        traceback.print_exc()
        return None

with Pool(args.proc_num) as p:
    rst = list(tqdm(p.imap(chat, need_list), total=len(need_list)))
