"""Dependency-light tests; no GPU or model weights are loaded."""
import ast
import collections
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prompt_contract as contract
from fsdp_prefix_index import indexed_apply_to_modules


def load_data_helpers():
    # Exercise the exact pure production functions without importing torch.
    tree = ast.parse((ROOT / "sft_data.py").read_text(encoding="utf-8"))
    names = {"completion_content", "prepare"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {"Any": Any, "collections": collections, "STOP_ID": 151669}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "sft_data.py", "exec"), namespace)
    return namespace


DATA = load_data_helpers()


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking and add_generation_prompt
        return [10, 11] if tokenize else "<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=False):
        result = []
        for i, part in enumerate(text.split("<stop>")):
            if i:
                result.append(151669)
            result.extend([7] * len(part))
        return result


class PromptTests(unittest.TestCase):
    def test_shared_prompt(self):
        messages = contract.canonical_messages("Question?", ["one", "two", "three", "four"])
        question, choices = contract.validate_messages(messages)
        self.assertEqual(question, "Question?")
        self.assertEqual(tuple(choices), tuple("ABCD"))
        self.assertEqual(contract.render_generation_prompt(FakeTokenizer(), messages),
                         "<|im_start|>assistant\n")

    def test_reject_different_system_prompt(self):
        messages = contract.canonical_messages("Question?", ["one", "two", "three", "four"])
        messages[0]["content"] = "A different prompt"
        with self.assertRaises(ValueError):
            contract.validate_messages(messages)

    def test_marker_is_not_final_answer(self):
        self.assertIsNone(contract.parse_strict_response("<think>Done.<stop>"))
        text = "<think>One point.<stop> Still thinking.</think><final_answer>A</final_answer>"
        self.assertEqual(contract.parse_strict_response(text)[1], "A")

    def test_strict_format(self):
        valid = "<think>Reason.</think><final_answer>A</final_answer>"
        self.assertIsNotNone(contract.parse_strict_response(valid))
        for invalid in [valid + "extra", valid + valid, valid.replace(">A<", ">a<"), "A"]:
            self.assertIsNone(contract.parse_strict_response(invalid))

    def test_chat_terminator_removed_once(self):
        self.assertEqual(contract.strip_generated_im_end("answer<|im_end|>"), "answer")

    def test_no_stop_in_question(self):
        with self.assertRaises(ValueError):
            contract.canonical_messages("Injected <stop>", ["a", "b", "c", "d"])


class CompletionTests(unittest.TestCase):
    def row(self):
        return {"prompt": [], "completion": [{"role": "assistant", "content": "x<stop>y"}],
                "stop_label_count": 1}

    def test_extract_content_not_container(self):
        self.assertEqual(DATA["completion_content"](self.row(), "train", 0), "x<stop>y")

    def test_reject_serialized_or_wrong_role(self):
        for completion in ["[{'role': 'assistant'}]", [{"role": "user", "content": "x"}], []]:
            row = self.row()
            row["completion"] = completion
            with self.assertRaises(RuntimeError):
                DATA["completion_content"](row, "train", 0)

    def test_mask_prompt_keep_atomic_label_and_eos(self):
        result, audit = DATA["prepare"]([self.row()], FakeTokenizer(), 20, "train")
        self.assertEqual(result[0]["labels"], [-100, -100, 7, 151669, 7, 2])
        self.assertEqual(audit["stop_labels"], 1)

    def test_no_silent_truncation(self):
        with self.assertRaises(RuntimeError):
            DATA["prepare"]([self.row()], FakeTokenizer(), 5, "train")

    def test_atomic_count_drift_fails(self):
        row = self.row()
        row["stop_label_count"] = 2
        with self.assertRaises(RuntimeError):
            DATA["prepare"]([row], FakeTokenizer(), 20, "train")


class PackagingTests(unittest.TestCase):
    def test_all_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_private_deployment_paths(self):
        private_paths = re.compile(r"/home/[\w.-]+/|[A-Z]:[\\/]Codex|/data/(?:experiment_data|corpora)_")
        for path in ROOT.rglob("*"):
            if path.suffix not in (".py", ".sh", ".md") or path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(private_paths.search(text), str(path))

    def test_runtime_paths_present(self):
        repo = ROOT.parents[1]
        for path in ("EarlyStop/verl/version/version", "vllm/__init__.py",
                     "training/medqa/runtime_overlay/verl/src/estar_a70_shared.py"):
            self.assertTrue((repo / path).is_file(), path)

    def test_fsdp_prefix_lookup(self):
        class Node:
            def __init__(self, children=()):
                self.children = children
            def named_children(self):
                return iter(self.children)
        root = Node([("module", Node([("layer", Node())]))])
        visited = []
        indexed_apply_to_modules(root, lambda m, p, d: visited.append((p, d)),
                                 lambda: None, ["layer.weight"])
        self.assertEqual(visited, [("", 0), ("", 1), ("layer.", 2)])


if __name__ == "__main__":
    unittest.main()
