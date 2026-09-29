import importlib
import torch.nn as nn

def load_external_module(spec):
    if not spec or ":" not in spec:
        raise NotImplementedError("A private HSA/HER module must be supplied as module:factory.")
    module_name, factory_name = spec.rsplit(":", 1)
    instance = getattr(importlib.import_module(module_name), factory_name)()
    if not isinstance(instance, nn.Module):
        raise TypeError("The private factory must return torch.nn.Module.")
    return instance
