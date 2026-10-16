"""Runs Python files with real PyTorch and prints what Atlas should report, as JSON.

Used by `npm run verify:torch` (scripts/verify-with-torch.mjs); it is optional and needs `pip install torch`.

    python scripts/torch_reference.py project <folder> <entry.py>
        Parameter count of every module called at the top level, plus the output shape of the model and
        of each of its direct children. Training scripts stop at the first optimizer.step(), like Atlas.
    python scripts/torch_reference.py cases <folder>
        For every <name>.py in the folder: parameter count and everything it printed.
"""
import contextlib
import io
import json
import os
import runpy
import sys
import traceback

import torch
import torch.nn as nn

torch.manual_seed(0)
roots, shapes, depth = [], {}, [0]
original_call = nn.Module.__call__


def shape_of(value):
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shape_of(item) for item in value]
    if isinstance(value, dict):
        return {key: shape_of(item) for key, item in value.items()}
    return None


def traced_call(self, *args, **kwargs):
    if depth[0] == 0 and not any(root is self for root in roots):
        roots.append(self)
    depth[0] += 1
    try:
        output = original_call(self, *args, **kwargs)
    finally:
        depth[0] -= 1
    shapes[id(self)] = shape_of(output)
    return output


class _Stop(BaseException):
    pass


def _stop_step(self, *args, **kwargs):
    raise _Stop()


nn.Module.__call__ = traced_call
for name in dir(torch.optim):
    optimizer = getattr(torch.optim, name)
    if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        optimizer.step = _stop_step


def run(path, folder):
    roots.clear()
    shapes.clear()
    output, error = io.StringIO(), None
    sys.path.insert(0, folder)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        with contextlib.redirect_stdout(output):
            runpy.run_path(path, run_name="__main__")
    except _Stop:
        pass
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        os.chdir(cwd)
        sys.path.remove(folder)
    parameters = {}
    for root in roots:
        for parameter in root.parameters():
            parameters[id(parameter)] = parameter.numel()
    return output.getvalue(), error, sum(parameters.values())


def main():
    mode, folder = sys.argv[1], os.path.abspath(sys.argv[2])
    if mode == "project":
        printed, error, total = run(os.path.join(folder, sys.argv[3]), folder)
        result = {"total": total, "error": error, "roots": [
            {"cls": type(root).__name__, "out": shapes.get(id(root)),
             "children": {name: shapes.get(id(child)) for name, child in root.named_children()}}
            for root in roots]}
    else:
        result = {}
        for file in sorted(os.listdir(folder)):
            if file.endswith(".py"):
                printed, error, total = run(os.path.join(folder, file), folder)
                result[file[:-3]] = {"total": total, "print": printed.strip(), "error": error}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
