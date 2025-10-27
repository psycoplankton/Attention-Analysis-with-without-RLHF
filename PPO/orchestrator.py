# Repository layout (Part 8 — RLHF with PPO)
#
#   PPO/
#     orchestrator.py          # run unit tests + optional tiny PPO demo
#     policy.py                # policy = SFT LM + value head (toy head on logits)
#     rollout.py               # prompt formatting, sampling, logprobs/KL utilities
#     ppo_loss.py              # PPO clipped objective + value + entropy + KL penalty
#     train_ppo.py             # single‑GPU RLHF loop (tiny, on‑policy)
#     eval_ppo.py              # compare reward vs. reference on a small set
#     tests/
#       test_ppo_loss.py
#       test_policy_forward.py
#
# Run from inside `PPO/`:
#   cd PPO
#   python orchestrator.py --demo 
#   pytest -q

import argparse, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(cmd.split(), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny PPO demo")
    args = p.parse_args()

    # 2) optional demo (requires SFT+RM checkpoints from Parts 6 & 7)
    if args.demo:
        # run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 10 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        # run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 50 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        run("python train_ppo.py --policy_ckpt ../SFT/runs/sft-demo/model_last.pt --reward_ckpt ../Reward_modelling/runs/rm-demo/model_last.pt --steps 100 --batch_size 4 --resp_len 128 --bpe_dir ../Scaling_Up/runs/part4-demo/tokenizer")
        run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../Reward_modelling/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../Scaling_Up/runs/part4-demo/tokenizer")

    print("\nPPO checks complete. ✅")