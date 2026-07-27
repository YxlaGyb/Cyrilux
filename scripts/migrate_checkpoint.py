"""迁移旧 checkpoint (state 7列) → 新格式 (state 9列 + conn_type).

Usage:
    python scripts/migrate_checkpoint.py out/model7/final.pt
    python scripts/migrate_checkpoint.py out/model7/final.pt -o out/model7/final_v2.pt
"""

import argparse, os, torch


def migrate(src: str, dst: str):
    print(f"加载: {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    pool = ckpt["pool"]

    # 1. state: [N, 7] → [N, 9]
    old_state = pool["state"]  # [N, 7] fp16
    N = old_state.shape[0]
    new_state = torch.zeros(N, 9, dtype=torch.float16)
    new_state[:, :7] = old_state  # 复制旧 7 列
    # BCM 默认值: slope=4.0, zero=0.25
    new_state[:, 7] = 4.0  # F_BCM_SLOPE
    new_state[:, 8] = 0.25  # F_BCM_ZERO
    pool["state"] = new_state
    print(f"  state: [N, 7] → [N, 9], BCM默认 slope=4.0 zero=0.25")

    # 2. conn_type: 所有旧突触标记为前馈 (0)
    if "syn_alive" in pool:
        S = pool["syn_alive"].shape[0]
        pool["conn_type"] = torch.zeros(S, dtype=torch.int8)
        alive = pool["syn_alive"]
        pool["conn_type"][alive] = 0  # feedforward
        print(f"  conn_type: 新增, {alive.sum().item()} 突触标记为 feedforward")

    # 3. 如果 _top_layer 是 7 (旧单层), 重映射到 10 (L4)
    if ckpt.get("top_layer") == 7:
        # 重映射 layer: 7 → LAYER_L4=10
        old_layer = pool["layer"]
        new_layer = old_layer.clone()
        new_layer[old_layer == 7] = 10
        pool["layer"] = new_layer
        ckpt["top_layer"] = 10
        print(f"  layer: 7 → 10 (L4)")

    ckpt["pool"] = pool
    ckpt["_migrated"] = True

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(ckpt, dst)
    print(f"保存: {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", help="旧 checkpoint 路径")
    p.add_argument("-o", "--output", help="输出路径 (默认: 同目录下 _v2.pt)")
    args = p.parse_args()
    dst = args.output or args.src.replace(".pt", "_v2.pt")
    migrate(args.src, dst)
