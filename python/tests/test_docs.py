"""Docstrings and documentation: present, matching the signatures, and runnable."""
import inspect
import os
import re

import numpy as np
import pytest

import blobboards

PUBLIC = [n for n in blobboards.__all__ if n != "PRINT_SIZES"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _section(doc, name):
    m = re.search(rf"\n\s*{name}\n\s*-+\n(.*?)(\n\s*\n\s*[A-Z][A-Za-z ]+\n\s*-+\n|\Z)", doc, re.S)
    return m.group(1) if m else ""


def _documented_params(doc):
    names = []
    for line in _section(doc, "Parameters").splitlines():
        m = re.match(r"^([A-Za-z_][\w, ]*?) :", line)     # `name : type` at the margin
        if m:
            names += [n.strip() for n in m.group(1).split(",")]
    return names


@pytest.mark.parametrize("name", PUBLIC)
def test_public_names_have_docstrings(name):
    obj = getattr(blobboards, name)
    assert inspect.getdoc(obj), f"{name} has no docstring"


@pytest.mark.parametrize("name", [n for n in PUBLIC if inspect.isfunction(getattr(blobboards, n))])
def test_documented_parameters_match_the_signature(name):
    fn = getattr(blobboards, name)
    documented = _documented_params(inspect.getdoc(fn))
    params = [p for p in inspect.signature(fn).parameters]
    assert set(documented) <= set(params), f"{name}: documented but not parameters: {set(documented) - set(params)}"
    assert set(params) <= set(documented), f"{name}: parameters not documented: {set(params) - set(documented)}"


def _examples(doc):
    code, lines = [], _section(doc, "Examples").splitlines()
    for line in lines:
        s = line.strip()
        if s.startswith(">>> ") or s.startswith("... "):
            code.append(s[4:])
        elif s in (">>>", "..."):
            code.append("")
    return "\n".join(code)


@pytest.mark.parametrize("name", PUBLIC + ["__module__"])
def test_docstring_examples_run(name, tmp_path, monkeypatch):
    obj = blobboards if name == "__module__" else getattr(blobboards, name)
    code = _examples(inspect.getdoc(obj) or "")
    if not code:
        pytest.skip("no example")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, f"<{name} example>", "exec"), {})


def _blocks(path):
    return re.findall(r"```python\n(.*?)```", open(path).read(), re.S)


@pytest.mark.parametrize("doc", ["docs/src/guide/python.md", "python/README.md"])
def test_documentation_examples_run(doc, tmp_path, monkeypatch, still_img):
    monkeypatch.chdir(tmp_path)
    import matplotlib
    matplotlib.use("Agg")
    ns = {"img": still_img}                       # the guide's "a grey image"
    for i, block in enumerate(_blocks(os.path.join(ROOT, doc))):
        exec(compile(block, f"<{doc} block {i}>", "exec"), ns)
