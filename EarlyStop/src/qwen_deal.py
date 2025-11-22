# condense_correct_think.py
import argparse
import json
import time
from typing import List, Dict

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import openai

# =========================
# Azure OpenAI 初始化
# =========================

# 方式一：用 key（与你示例一致）
AZURE_OPENAI_SUBSCRIPTION_KEY = "0cd189cfe5484af88ad4f0b992cf24fc"

def new_azure_openai_client(ws_name: str, api_version: str = "2024-12-01-preview") -> openai.AzureOpenAI:
    """
    用 API Key 初始化 Azure OpenAI 客户端
    """
    endpoint = f"https://{ws_name}.openai.azure.com/"  # 修正了示例里的 endpoint
    client = openai.AzureOpenAI(
        api_key=AZURE_OPENAI_SUBSCRIPTION_KEY,
        azure_endpoint=endpoint,
        api_version=api_version,
        timeout=80.0,
        max_retries=3,
    )
    return client

# 方式二：用 AAD（可选；如需改用，取消注释并在 main 里替换）
def new_azure_openai_client_with_aad(ws_name: str, api_version: str = "2024-12-01-preview") -> openai.AzureOpenAI:
    endpoint = f"https://{ws_name}.openai.azure.com/"
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = openai.AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=endpoint,
        api_version=api_version,
        timeout=80.0,
        max_retries=3,
    )
    return client

# =========================
# GPT 调用
# =========================

SYSTEM_BRIEFEN = (
    "You are a concise rewriting assistant. For the given segment, return ONLY a very short phrase "
    "or a few comma-separated keywords (<= 10 words total). "
    "If the original segment contains the token 'Wait' (case sensitive), you MUST keep the token 'Wait' in your output. "
    "No quotes, no extra text."
)


def condense_segment(client: openai.AzureOpenAI, deployment_id: str, segment: str, temperature: float = 0.2) -> str:
    """
    把一段文本“浓缩成简短短语或关键词”并返回。
    只返回模型的一行文字（去掉首尾空白）。
    """
    # 防御性截断（可选）：防止超长
    segment = segment.strip()
    if not segment:
        return ""

    for attempt in range(4):  # 简单重试
        out = ''
        try:
            resp = client.chat.completions.create(
                model=deployment_id,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_BRIEFEN},
                    {"role": "user", "content": f"Segment: {segment}\nReturn only concise phrase or keywords."}
                ]
            )
            out = resp.choices[0].message.content.strip()
            # 保证单行输出
            out = " ".join(out.splitlines()).strip()
            return out
        except Exception as e:
            # 简单退避
            wait = 1.5 * (attempt + 1)
            print(f"[warn] condense failed (attempt {attempt+1}): {e}. sleep {wait:.1f}s {out}")
            time.sleep(wait)
    # 多次失败就返回原文的精简片段（兜底）
    return segment

# =========================
# 处理 correct_think
# =========================

def split_think(think: str) -> List[str]:
    # 按 '->' 切分，清理空白
    parts = [p.strip() for p in think.split("->")]
    return [p for p in parts if p]  # 过滤空

def process_item_correct_think(client: openai.AzureOpenAI, deployment_id: str, item: Dict) -> bool:
    """
    处理单个样本，若存在 'correct_think' 就更新为压缩后的版本。
    返回是否有更新。
    """
    think = item.get("correct_think")
    if not isinstance(think, str) or not think.strip():
        return False

    segments = split_think(think)
    if not segments:
        return False

    condensed_segments = []
    for i, seg in enumerate(segments, 1):
        condensed = condense_segment(client, deployment_id, seg)
        condensed_segments.append(condensed)
        # 进度提示
        if i % 10 == 0 or i == len(segments):
            print(f"  - condensed {i}/{len(segments)} segments")

    new_think = " -> ".join(condensed_segments)
    item["correct_think"] = new_think
    return True

def process_file(
    client: openai.AzureOpenAI,
    deployment_id: str,
    in_path: str,
    out_path: str,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是列表。")

    updated = 0
    total = len(data)
    print(f"[info] loaded {total} items from {in_path}")

    for idx, item in enumerate(data, 1):
        if process_item_correct_think(client, deployment_id, item):
            updated += 1
        if idx % 20 == 0 or idx == total:
            print(f"[info] processed {idx}/{total} items (updated: {updated})")

    with open(out_path, "w", encoding="utf-8") as f:
        # 保持可读性；如果你想“无缩进”，把 indent=None 并设置 separators=(',', ':')
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[done] updated {updated}/{total} items with condensed correct_think -> saved to {out_path}")

# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Condense 'correct_think' by calling Azure OpenAI (gpt-4o-mini).")
    parser.add_argument("--input", type=str, default="/home/azureuser/cloudfiles/code/Users/junda.wang/project/AdaptThink/data/train/medicalqa/tmp_output_json/")
    parser.add_argument("--output", type=str, default="/home/azureuser/cloudfiles/code/Users/junda.wang/project/AdaptThink/data/train/medicalqa/medicalqa_with_correct_think2.json")
    parser.add_argument("--ws-name", default="cse51muz9j6fgvfopenai", help="Azure OpenAI 资源名称（workspace name）")
    parser.add_argument("--deployment-id", default="gpt-4o-mini", help="gpt4omini 的部署名（你在 Azure 上的 deployment 名称）")
    parser.add_argument("--api-version", default="2024-12-01-preview", help="Azure OpenAI API 版本")
    parser.add_argument("--use-aad", action="store_true", help="使用 AAD 方式鉴权（默认用 API Key）")
    args = parser.parse_args()

    out_path = args.output or args.input.replace(".json", "_condensed.json")

    # 选择鉴权方式
    if args.use_aad:
        client = new_azure_openai_client_with_aad(args.ws_name, args.api_version)
    else:
        client = new_azure_openai_client(args.ws_name, args.api_version)

    process_file(client, args.deployment_id, args.input, out_path)

if __name__ == "__main__":
    main()




