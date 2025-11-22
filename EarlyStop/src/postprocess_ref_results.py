
import json, jsonlines
from tqdm import tqdm
from adapt_think_rm import adapt_think_rm
from multiprocessing import Pool
import os
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/zjj/hf/DeepSeek-R1-Distill-Qwen-7B") 
    parser.add_argument("--input_path", type=str, default="/mnt/zjj/open-source/github/AdaptThink/adapt_think/data/train/ref_presampling/DeepSeek-R1-Distill-Qwen-1.5B_deepscaler_n10_K16_len16384.jsonl")
    parser.add_argument("--output_path", type=str, default="/mnt/zjj/open-source/github/AdaptThink/adapt_think/data/train/ref_results/DeepSeek-R1-Distill-Qwen-1.5B_deepscaler_K16_len16384.json")
    parser.add_argument("--answer_key", type=str, default="answer")
    parser.add_argument("--nothinking", action='store_true', default=False)   
    return parser.parse_args()

args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

data = defaultdict(list)
for js in tqdm(jsonlines.open(args.input_path,"r")):
    problem = js['problem'].strip()
    data[problem].append(js)

print("num data:", len(data))

def process(problem):
    rst = []
    items = data[problem]
    assert problem == items[0]['problem'].strip()
    real_answer = items[0][args.answer_key]
    solutions, correctness, lengths, truncates = {}, {}, {}, {}
    truncates = [(item['response']['choices'][0]['finish_reason'] == 'length') for item in items]
    solutions = [(('</think>' if args.nothinking else '') + item['response']['choices'][0]['text']) for item in items]
    lengths = [((args.nothinking == True) + item['response']['usage']['completion_tokens']) for item in items]
    correctness = [adapt_think_rm(data_source='', solution_str=solution, ground_truth=real_answer)['acc'] for solution in solutions]
    
    avg_acc = np.mean(correctness)
    avg_len = np.mean(lengths)
    avg_clip_ratio = np.mean(truncates)
    # sometimes tokenizer would change some special tokens
    processed_problem = tokenizer.decode(tokenizer.encode(problem, add_special_tokens=False))
    return {
        'problem': processed_problem,
        'answer': real_answer,
        'metrics': {
            'n_responses': len(items),
            'avg_acc_thinking': avg_acc,
            'avg_len_thinking': avg_len,
            'avg_clip_ratio': avg_clip_ratio,
        }
    }
    

with Pool(64) as p:
    results = list(tqdm(p.imap(process, list(data.keys())), total=len(data)))

overall_metrics = {}
for key in results[0]['metrics']:
    overall_metrics[key] = np.mean([js['metrics'][key] for js in results])
results = [{'problem': '__OVERALL__', 'answer': None, 'metrics': overall_metrics}] + results
for key, value in overall_metrics.items():
    print(f'{key}: {value}')
save_dir = args.output_path.rsplit('/', 1)[0]
os.makedirs(save_dir, exist_ok=True)
json.dump(results, open(args.output_path, 'w'), indent=2, ensure_ascii=False)
    

