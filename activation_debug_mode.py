import torch
import torch.nn as nn
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import TorchDispatchMode
import weakref
from collections import defaultdict

from torchvision.models import resnet18

parents = ['Global']
tensor_meta = {}
tensor_id = 0
alive_tensors = weakref.WeakValueDictionary()

def count_alive_tensors():
    count = defaultdict(int)
    for id, val in alive_tensors.items():
        count[tensor_meta[id]] += val.storage().nbytes()

    for k, v in count.items():
        if v > 100:
            print(k, f"{v/1e6:.2f} MB")


class ActivationDebugMode(TorchDispatchMode):
    def __init__(self, verbose=False):
        self.verbose: bool = verbose

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        rs = func(*args, **kwargs)
        def track_output(val):
            if isinstance(val, torch.Tensor):
                global tensor_id
                tensor_meta[tensor_id] = tuple(parents)
                alive_tensors[tensor_id] = val
                tensor_id += 1

        tree_map(track_output, rs)

        if func == torch.ops.aten.detach.default:
            return rs
        return rs

val = torch.randn(64, 128, 1024)

def create_backwards_push(name):
    class PushState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
            if len(args) == 1:
                return args[0]
            return args

        @staticmethod
        def backward(ctx, *grad_outs):
            global parents
            parents.append(name)
            return grad_outs

    return PushState.apply

def create_backwards_pop(name):
    class PopState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
            if len(args) == 1:
                return args[0]
            return args

        @staticmethod
        def backward(ctx, *grad_outs):
            global parents
            assert(parents[-1] == name)
            print()
            print(f"Activations saved at {'.'.join(parents)} backwards end")
            count_alive_tensors()
            parents.pop()
            return grad_outs

    return PopState.apply

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x

def enter_module(name):
    def f(module, inputs):
        global parents
        parents.append(name)
        inputs = normalize_tuple(inputs)
        out = create_backwards_pop(name)(*inputs)
        return out

    return f

def exit_module(name):
    def f(module, inputs, outputs):
        global parents
        assert(parents[-1] == name)
        parents.pop()
        outputs = normalize_tuple(outputs)
        return create_backwards_push(name)(*outputs)
    return f

def instrument_module(mod):
    for name, module in dict(mod.named_children()).items():
        module.register_forward_pre_hook(enter_module(name))
        module.register_forward_hook(exit_module(name))
