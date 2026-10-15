"""Rewrite Pad(mode=wrap|edge) nodes as Slice+Concat so the ORT CUDA EP can
claim them. Semantics are identical; only the op decomposition changes.

ONNXRuntime's CUDA EP has no kernel for Pad in wrap/edge mode, so every such
node fell back to the CPU EP, shredding the descriptor graph into CPU/GPU
segments with a device copy at each boundary (16 Memcpy nodes on
track_logpolar_circ_s96.onnx). Slice and Concat *are* CUDA-supported, so the
decomposition below leaves the whole graph on the GPU. See
results/descriptor_inference_speedup.md.

Usage (onnx is not in any pixi env; a throwaway uv venv is enough):

    uv venv /tmp/onnxenv && uv pip install --python /tmp/onnxenv/bin/python onnx
    /tmp/onnxenv/bin/python experiments/onnx_depad.py in.onnx out.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

INT_MAX = 9223372036854775807

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
init = {i.name: numpy_helper.to_array(i) for i in g.initializer}

_const_cache = {}


def const1d(vals):
    """Interned 1-D int64 initializer."""
    key = tuple(vals)
    if key not in _const_cache:
        name = f"depad_c_{'_'.join(str(v).replace('-', 'm') for v in vals)}"
        g.initializer.append(numpy_helper.from_array(
            np.asarray(vals, dtype=np.int64), name))
        _const_cache[key] = name
    return _const_cache[key]


def slice_node(inp, out, starts, ends, axes):
    return helper.make_node(
        "Slice", [inp, const1d(starts), const1d(ends), const1d(axes)], [out],
        name=out)


new_nodes = []
rewritten = 0
for n in g.node:
    if n.op_type != "Pad":
        new_nodes.append(n)
        continue
    mode = next((helper.get_attribute_value(a).decode()
                 for a in n.attribute if a.name == "mode"), "constant")
    if mode not in ("wrap", "edge"):
        new_nodes.append(n)
        continue
    if len(n.input) < 2 or n.input[1] not in init:
        raise RuntimeError(f"{n.name}: non-constant pads, cannot rewrite")
    if len(n.input) > 2 and n.input[2]:
        raise RuntimeError(f"{n.name}: unexpected constant_value input")
    pads = init[n.input[1]].tolist()
    rank = len(pads) // 2
    axes = [a for a in range(rank) if pads[a] or pads[a + rank]]
    if len(axes) != 1:
        raise RuntimeError(f"{n.name}: expected a single padded axis, got {axes}")
    ax = axes[0]
    b, e = pads[ax], pads[ax + rank]
    x, y = n.input[0], n.output[0]
    tag = f"depad_{n.name}"
    parts = []
    if b:
        pre = f"{tag}_pre"
        if mode == "wrap":
            new_nodes.append(slice_node(x, pre, [-b], [INT_MAX], [ax]))
        else:  # edge: replicate the first row b times
            edge = f"{tag}_pre1"
            new_nodes.append(slice_node(x, edge, [0], [1], [ax]))
            if b == 1:
                pre = edge
            else:
                new_nodes.append(helper.make_node(
                    "Concat", [edge] * b, [pre], name=pre, axis=ax))
        parts.append(pre)
    parts.append(x)
    if e:
        post = f"{tag}_post"
        if mode == "wrap":
            new_nodes.append(slice_node(x, post, [0], [e], [ax]))
        else:
            edge = f"{tag}_post1"
            new_nodes.append(slice_node(x, edge, [-1], [INT_MAX], [ax]))
            if e == 1:
                post = edge
            else:
                new_nodes.append(helper.make_node(
                    "Concat", [edge] * e, [post], name=post, axis=ax))
        parts.append(post)
    new_nodes.append(helper.make_node("Concat", parts, [y], name=tag, axis=ax))
    rewritten += 1
    print(f"rewrote {n.name:16s} mode={mode:5s} axis={ax} pads=({b},{e})")

del g.node[:]
g.node.extend(new_nodes)

# Drop initializers that no longer feed anything.
used = {i for n in g.node for i in n.input}
keep = [i for i in g.initializer if i.name in used]
dropped = len(g.initializer) - len(keep)
del g.initializer[:]
g.initializer.extend(keep)

m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
onnx.checker.check_model(m, full_check=True)
onnx.save(m, dst)
print(f"\nrewrote {rewritten} Pad nodes; dropped {dropped} dead initializers")
print(f"nodes {len(g.node)}, initializers {len(g.initializer)} -> {dst}")
