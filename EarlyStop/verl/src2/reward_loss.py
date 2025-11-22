import torch
import torch.nn.functional as F
import re

import unicodedata

FINAL_TAG_RE  = re.compile(r"<final_answer>\s*([A-D])\s*</final_answer>", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
THINK_RE      = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


# ======== 工具函数 ========
def _strip_tags(s: str) -> str:
    """去掉 <think> / </think> / <final_answer>…</final_answer> 标签本身（不动 <answer>，因为它们单独处理）"""
    s = re.sub(r"</?think>", "", s or "", flags=re.IGNORECASE)
    s = re.sub(r"<final_answer>.*?</final_answer>", "", s, flags=re.IGNORECASE | re.DOTALL)
    return s


def _remove_answer_tags(s: str) -> str:
    """去掉 <answer>…</answer>（用于计算“可见长度”）"""
    return ANSWER_TAG_RE.sub("", s or "")


def _get_final_letter_or_gold(solution_str: str, gold: str) -> str | None:
    """优先从 <final_answer> 里取 A-D；没有就用 gold 的首字母（若是 A-D）"""
    m = FINAL_TAG_RE.search(solution_str or "")
    if m:
        return m.group(1).upper()
    if isinstance(gold, str) and gold:
        ch = gold.strip()[0].upper()
        if ch in "ABCD":
            return ch
    return None


def _extract_answer_matches(solution_str: str):
    """返回所有 <answer>…</answer> 的 Match 对象列表"""
    return list(ANSWER_TAG_RE.finditer(solution_str or ""))


def _get_think_span(solution_str: str):
    """返回 (start_idx, end_idx, think_body)；若无 think 则返回 (None, None, None)"""
    m = THINK_RE.search(solution_str or "")
    if not m:
        return None, None, None
    return m.start(1), m.end(1), m.group(1)


def _weighted_answer_score(solution_str: str, gold: str) -> tuple[float, int]:
    """
    找到所有 <answer>X</answer>，计算“越前面越高”的权重 w = 1 - l/L，
      - l: 该标签之前的“可见字符”数（去掉 <think>/<final_answer> 以及任意 <answer>… 块）
      - L: think 内（若存在）或全文的“可见字符总长”
    与目标字母（<final_answer> 优先，否则 gold 首字母）一致记 w，否则记 0。
    返回 (加权平均得分 in [0,1], 计入的标签总数)；若无标签则 (0.0, 0)。
    """
    target = _get_final_letter_or_gold(solution_str, gold)
    if target is None:
        return 0.0, 0


    matches = _extract_answer_matches(solution_str)
    if not matches:
        return 0.0, 0


    # 在 <think> 内做位置度量；没有 think 就对全文
    t_start, t_end, think_body = _get_think_span(solution_str)
    if think_body is not None:
        visible_all = _strip_tags(_remove_answer_tags(think_body))
        L = len(visible_all)
        base_slice_start = t_start  # 计算“pre 可见长度”时的切片起点
    else:
        # 全文可见长度
        visible_all = _strip_tags(_remove_answer_tags(solution_str))
        L = len(visible_all)
        base_slice_start = 0


    if L <= 0:
        # 没有可视字符，退化为等权平均
        correct = sum(1 for m in matches if (m.group(1).upper() == target))
        return (correct / len(matches)), len(matches)


    total_w, total_w_correct = 0.0, 0.0
    for m in matches:
        # 只用落在 think 内的标签做“位置权重”；不在 think 内的按最小权重处理
        if think_body is not None and not (t_start <= m.start() <= t_end):
            w = 0.0  # 如果你也想计入，可改成一个很小的常数权重
        else:
            pre = solution_str[base_slice_start:m.start()]
            l = len(_strip_tags(_remove_answer_tags(pre)))
            w = max(0.0, min(1.0, 1.0 - (l / L)))


        total_w += w
        if m.group(1).upper() == target:
            total_w_correct += w


    if total_w <= 0.0:
        # 没有有效权重则退化为等权
        correct = sum(1 for m in matches if (m.group(1).upper() == target))
        return (correct / len(matches)), len(matches)


    return (total_w_correct / total_w), len(matches)

def contains_non_english_letters(txt: str) -> bool:
    """
    若文本中存在“不是 ASCII 英文字母(A–Z/a–z)的 Unicode 字母”，返回 True。
    数字/标点/空白/emoji 等不计入“字母”范畴，不影响结果。
    """
    for ch in txt:
        # 只关注“字母”类字符（Unicode 类别以 'L' 开头）
        if unicodedata.category(ch).startswith("L"):
            if not ("A" <= ch <= "Z" or "a" <= ch <= "z"):
                return True
    return False

def length_match_score(len_steps: int, think_len: float, logit_coef: float) -> float:
    """
    根据 len_steps 与目标 target = think_len * logit_coef 的相对位置给分：
      - 若 logit_coef < 0.5:
          len_steps <= target => 1
          len_steps  > target => target / len_steps
      - 若 logit_coef >= 0.5:
          len_steps >= target => 1
          len_steps  < target => len_steps / target
    返回值 ∈ [0, 1]。
    """
    # 计算目标
    target = float(think_len) * float(logit_coef)


    # 边界：target <= 0 的情况
    if target <= 0.0:
        if logit_coef < 0.5:
            return 1.0 if len_steps <= 0 else 0.0
        else:
            # 规则上：len_steps >= target(=0) 时记 1
            return 1.0


    ratio = float(len_steps) / target


    if logit_coef < 0.5:
        score = 1.0 if len_steps <= target else (target / float(len_steps))
    else:
        score = 1.0 if len_steps >= target else ratio


    # 保底到 [0,1]
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
    return float(score)








def compute_score(
    data_source,               # 数据源，可以用于条件奖励
    solution_str,              # 生成的完整response文本
    ground_truth,              # 标准答案
    #split_scores=0.0,
    #stop_resp_scores,
    #stop_count,
    hard_prob=1.0,
    extra_info=None,           # 可选，包含其它元数据
    tokenizer=None,            # 必须传入tokenizer
    output_logits=None,        # 可选，模型生成时的logits（如有，可支持tag reward）
    options=None,              # 可选，选项字典，如 {'A': 'xx', ...}
    think_len=60,             # 标准步骤长度
    penalty_lambda=1.0         # 可调惩罚系数
):
    # === 1. 判定准确率 ===
    # 你可以定制选项解析或自由文本
    
    is_correct = False
    pred_answer = None
    gold_answer = ground_truth
    if options:
        # 用 pattern 找 <answer> 标签里的内容
        match = re.search(r'<final_answer>(.*?)</final_answer>', solution_str, re.DOTALL)
        if match:
            parsed_answer = match.group(1).strip()
            pred_answer = get_choice(parsed_answer, options)
            is_correct = gold_answer == pred_answer
        else:
            is_correct = False
    else:
        m = FINAL_TAG_RE.search(solution_str)
        if m:
            parsed_answer = m.group(1).upper()
            gt0 = str(ground_truth).strip()[0].upper() if isinstance(ground_truth, str) and ground_truth else ""
            is_correct = (parsed_answer == gt0)
            pred_answer = parsed_answer
        else:
            is_correct = False

    answer_tag_score, answer_tag_count = _weighted_answer_score(solution_str, ground_truth)
    """
    Your code here
    """
    # === 2. 解析 think steps ===
    think_steps = split_think_steps(solution_str)
    # === 3. 解析 logits tag（可选，没有就默认1.0） ===
    logit_coef = hard_prob
    # === 4. 计算格式奖励 ===
    combine_think_steps = '\n'.join(think_steps)
    format_reward = 1
    #format_reward = 1
    if combine_think_steps.count("<Hard>") == 1 or combine_think_steps.count("<Easy>") == 1:
        format_reward = 1
    format_reward3 = 0
    format_reward4 = 0
    format_reward5 = 0

    if solution_str.count('<final_answer>') == 1 and solution_str.count('</final_answer>') == 1:
        format_reward3 += 0.5
        if len(solution_str.split('</final_answer>')[1]) < 10:
            format_reward4 = 1
    if solution_str.count('<think>') == 1 and solution_str.count('</think>') == 1:
        format_reward3 += 0.5
    if solution_str.count('<stop>') > 0 and solution_str.count('<stopr>') < 0:
        format_reward5 = 1
    
    format_reward = format_reward / 2 + format_reward5 / 2
    format_reward2 = format_reward3 / 2 + format_reward4 / 2
    if contains_non_english_letters(solution_str) or re.search(r' {10,}', solution_str):
        format_reward = 0
        format_reward2 = 0
    
    accuracy_reward = 1 if is_correct else 0
    total_reward = (
        #length_match_score(len(think_steps), think_len, logit_coef) * 1/8 +
        format_reward * 1/8 +
        answer_tag_score * 1/8 +
        accuracy_reward * 1/8 +
        format_reward2 * 1/2
    )
    print(f"stop_resp_scores:{answer_tag_score:.3f} format_reward:{format_reward:.3f} format_reward2:{format_reward2:.3f} accuracy_reward:{accuracy_reward} stop_num:{answer_tag_count:.3f}")
    #print(f"format_reward2:{format_reward2:.3f} accuracy_reward:{accuracy_reward}")
    # === 9. 返回 dict（全字段可扩展） ===
    return {
        "score": total_reward,
        "acc": accuracy_reward,
        "pred": pred_answer,
    }

# --- 辅助函数（建议保持你的实现） ---

def get_choice(answer: str, options: dict):
    from thefuzz import fuzz
    highest_score = -1
    highest_option = None
    for char, opt in options.items():
        option = char + ": " + opt
        score = max(fuzz.ratio(answer, option), fuzz.ratio(answer, char), fuzz.ratio(answer, opt))
        if score > highest_score:
            highest_score = score
            highest_option = char
    return highest_option

def split_think_steps(think_chain):
    steps = re.split(r'(?:->|[.!?\n])', think_chain)
    return [s.strip() for s in steps if s.strip()]

def get_first_tag_logits(output_logits, completion_text, tokenizer):
    match = re.search(r'<(Hard|Easy)>', completion_text)
    if not match:
        return 1.0
    tag_str = "<" + match.group(1) + ">"
    tag_ids = tokenizer.encode(tag_str, add_special_tokens=False)
    encoding = tokenizer(completion_text, add_special_tokens=False)
    input_ids = encoding['input_ids']
    tag_start_idx = None
    for i in range(len(input_ids) - len(tag_ids) + 1):
        if input_ids[i:i + len(tag_ids)-1] == tag_ids[:-1]:
            tag_start_idx = i
            break
    if tag_start_idx is None:
        return 1.0
    hard_id = tokenizer.convert_tokens_to_ids('Hard')
    easy_id = tokenizer.convert_tokens_to_ids('Easy')
    logits = output_logits[tag_start_idx + 1]
    probs = F.softmax(logits, dim=-1)
    prob_hard = probs[hard_id].item()
    prob_easy = probs[easy_id].item()
    denom = prob_hard + prob_easy
    if denom == 0:
        return 1.0
    prob = prob_hard / denom
    return prob
