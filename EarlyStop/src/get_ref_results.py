import json
from tqdm import tqdm

ipt_path = '/mnt/zjj/dev/fast_think/verl/recipe/fast_think/data/train/ref_results/DeepSeek-R1-Distill-Qwen-7B_deepscaler_K16_sl16384_fl4096.json'
opt_path = '/mnt/zjj/open-source/github/AdaptThink/adapt_think/data/train/ref_results/DeepSeek-R1-Distill-Qwen-7B_deepscaler_K16_len16384.json'
data = []
for js in tqdm(json.load(open(ipt_path, 'r'))):
    data.append({
        'problem': js['problem'],
        'answer': js['answer'],
        'metrics': {
            'n_responses': js['metrics']['n_slow'],
            'avg_acc_thinking': js['metrics']['avg_acc_slow'],
            'avg_len_thinking': js['metrics']['avg_len_slow'],
            'avg_clip_ratio': js['metrics']['avg_clip_ratio_slow'],
        }
    })

json.dump(data, open(opt_path, 'w'), indent=2, ensure_ascii=False)

