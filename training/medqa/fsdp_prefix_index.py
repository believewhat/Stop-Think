"""Equivalent prefix lookup for PyTorch 2.7 FSDP metadata traversal.

The upstream helper scans every parameter FQN at every module. Sparse MoE
with thousands of PEFT wrappers makes that one-time scan quadratic. Indexing
all dotted prefixes preserves its wrapper-prefix fallback without that scan.
No tensor, gradient, communication or optimizer operation is changed.
"""
import inspect


def indexed_apply_to_modules(root_module, module_fn, return_fn, filter_fqns=None, *args, **kwargs):
    prefixes = None
    if filter_fqns is not None:
        prefixes = set()
        for fqn in filter_fqns:
            for i, char in enumerate(fqn):
                if char == '.':
                    prefixes.add(fqn[:i + 1])

    def walk(module, prefix, tree_level):
        module_fn(module, prefix, tree_level, *args, **kwargs)
        for name, child in module.named_children():
            if child is None:
                continue
            new_prefix = prefix + name + '.'
            if prefixes is not None and new_prefix not in prefixes:
                if name in ('_fsdp_wrapped_module', '_dmp_wrapped_module', 'module'):
                    new_prefix = prefix
            walk(child, new_prefix, tree_level + 1)

    walk(root_module, '', 0)
    return return_fn(*args, **kwargs)


def install_fsdp_prefix_index():
    import torch
    from torch.distributed.fsdp import _common_utils
    if _common_utils._apply_to_modules is indexed_apply_to_modules:
        return
    assert torch.__version__.split('+')[0] == '2.7.0', 'Revalidate traversal shim before upgrading torch'
    source = inspect.getsource(_common_utils._apply_to_modules)
    assert 'for fqn in filter_fqns:' in source and 'if fqn.startswith(new_prefix):' in source
    _common_utils._apply_to_modules = indexed_apply_to_modules
    print('FSDP_PREFIX_INDEX_ENABLED: equivalent dotted-prefix metadata lookup', flush=True)
