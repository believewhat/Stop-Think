#!/usr/bin/env python
# -*- coding: utf-8 -*-





import os
import argparse
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="如 Qwen/Qwen3-8B 或本地权重路径")
    p.add_argument("--train_file", type=str, required=True,
                   help="训练集 JSON/JSONL（会话式 prompt+completion）")
    p.add_argument("--eval_file", type=str, default=None,
                   help="可选：验证集 JSON/JSONL（同格式）")
    p.add_argument("--output_dir", type=str, default="sft_out")


    # 训练超参
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_length", type=int, default=20000, help="样本拼接后的最大长度（prompt+completion）")


    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default=None,
                   help="DeepSpeed 配置 JSON 路径（如 ds_zero3.json）")
    return p.parse_args()




def main():
    args = parse_args()



    # 加载数据（保持你现有的会话式 prompt+completion 格式）
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = None
    if args.eval_file:
        eval_ds = load_dataset("json", data_files=args.eval_file, split="train")


    # 训练配置（与官方文档一致；不传 tokenizer；让 TRL 自动应用 chat template）
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        packing=args.packing,                 # 是否启用 packing
        bf16=args.bf16,                       # Qwen 等模型推荐 bfloat16
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=[],
        completion_only_loss=True,
        # 关键点：prompt-completion 数据集默认只训 completion（官方默认行为）
        # 如果你想显式声明（部分版本支持）：
        # completion_only_loss=True,
        # 另外：若需要指定 from_pretrained 的 dtype/trust_remote_code 等，可加：
        model_init_kwargs={
            "torch_dtype": "auto",
            "trust_remote_code": True,
        },
        deepspeed=args.deepspeed,
    )


    # 直接把 model 传 **字符串**（推荐做法，Trainer 会自行加载模型和分词器）
    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # 不传 tokenizer；不自建 template；让 TRL 根据模型 chat_template 自动处理
    )


    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()






