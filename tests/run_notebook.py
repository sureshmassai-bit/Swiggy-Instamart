"""Execute a notebook's code cells top to bottom in ONE fresh Python process (a stand-in for
"Restart & Run All" when Jupyter/nbclient is not installed).

    python tests/run_notebook.py notebooks/Customer_Intelligence_Support_Copilot.ipynb

It is not a Jupyter kernel: it provides a `display()` shim and does not capture rich outputs or
render magics. It does prove every cell runs without error, in order, from a clean interpreter.
"""
import ast
import builtins
import json
import sys
import time
import traceback


def main(path):
    nb = json.load(open(path, encoding="utf-8"))
    assert nb["nbformat"] == 4, "expected nbformat 4"
    builtins.display = lambda *objs: [print(o) for o in objs]
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        tree = ast.parse(src, filename=f"cell[{i}]")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        t0 = time.time()
        try:
            exec(compile(tree, f"cell[{i}]", "exec"), ns)
            if last is not None:                                  # echo the final expression like Jupyter
                val = eval(compile(ast.Expression(last.value), f"cell[{i}]", "eval"), ns)
                if val is not None:
                    print(val)
        except Exception:
            print(f"\nCELL {i} FAILED:\n{src}\n", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"--- cell {i} ok ({time.time() - t0:.1f}s)")
    print("NOTEBOOK RAN TOP TO BOTTOM WITHOUT ERRORS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
