# Repository layout (Part 5)
#
#   part_5/
#     orchestrator.py            # run unit tests + optional MoE demo
#     README.md                  # 5.1/5.3 concept notes (compact MD)
#     gating.py                  # router/gating (top‑k) + load‑balancing aux loss
#     experts.py                 # MLP experts (SwiGLU or GELU)
#     moe.py                     # Mixture-of-Experts layer (dispatch/combine)
#     block_hybrid.py            # Hybrid dense+MoE block examples
#     demo_moe.py                # small forward pass demo + routing histogram
#     tests/
#       test_gate_shapes.py
#       test_moe_forward.py
#       test_hybrid_block.py
#
# Run from inside `part_5/`:
#   cd part_5
#   python orchestrator.py --demo
#   pytest -q

import argparse, pathlib, subprocess, sys, shlex

ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run a tiny MoE demo")
    args = p.parse_args()

    # 2) optional demo
    if args.demo:
        run("python demo.py --tokens 6 --hidden 128 --experts 4 --top_k 1")

    print("\nMOE checks complete. ✅")