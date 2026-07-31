"""PPA 模型评估 — 世界模型内在自洽 (自由能), 表示分化, 生成连贯性.

不做 PPL/Top-1 (用户裁决: 已弃用). 指标:
- 自由能 (L5→L4 主误差 + 各层预测误差, 经精度加权)
- cos(z5): 不同输入表示是否分化
- 生成: W_future 时空共振自顶向下重建字节, ASCII 比例/连贯性
- Memory: 重复喂同输入, 自由能是否逐轮下降 (世界模型记住模式)
"""
import time
import torch
from model.pc.dense.core import DensePCNet, DensePCConfig

torch.set_grad_enabled(False)


def load_model(path):
    t0 = time.perf_counter()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    d_l4 = sd["W_04"].shape[0]
    d_l2 = sd["W_42"].shape[0]
    d_l3 = sd["W_23"].shape[0]
    d_l5 = sd["W_35"].shape[0]
    d_l6 = sd["W_56"].shape[0]
    cfg = DensePCConfig(d_l4=d_l4, d_l2=d_l2, d_l3=d_l3,
                        d_l5=d_l5, d_l6=d_l6, max_seq_len=256)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DensePCNet(cfg).to(dev)
    net.load_state_dict(sd, strict=False)
    net.active_size = {
        "l4": net.W_04.shape[0], "l2": net.W_42.shape[0],
        "l3": net.W_23.shape[0], "l5": net.W_35.shape[0], "l6": net.W_56.shape[0],
    }
    for k in ("l4", "l2", "l3", "l5", "l6"):
        a = net.active_size[k]
        net._death_row[k] = torch.zeros(a, dtype=torch.int8, device=dev)
        net._probation_counter[k] = torch.zeros(a, dtype=torch.int16, device=dev)
    elapsed = time.perf_counter() - t0
    d = cfg.dims()
    print(f"  {path}: L4={d['l4']} L2={d['l2']} L3={d['l3']} "
          f"L5={d['l5']} L6={d['l6']}  params={cfg.param_count():,}  load={elapsed:.1f}s")
    print(f"    active_size: l4={net.active_size['l4']} l2={net.active_size['l2']} "
          f"l3={net.active_size['l3']} l5={net.active_size['l5']} l6={net.active_size['l6']}")
    return net, cfg


def _byte_tensor(text: str, dev) -> torch.Tensor:
    return torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=dev)


def test_free_energy(net, texts=None):
    """各输入的自由能 (世界模型预测难度)."""
    if texts is None:
        texts = [
            "Hello world! How are you today?",
            "import torch; def foo(x): return x*2",
            "123 + 456 = 579, 789 * 10 = 7890",
            "人工智能的未来在于",
        ]
    print("\n  [Free Energy / Selectivity]")
    dev = next(net.parameters()).device
    results = {}
    for text in texts:
        bv = _byte_tensor(text, dev)
        out = net(bv)
        results[text[:20]] = net._z5[0, -1].float().clone()
        print(f"    {text[:30]:32s}  FE={float(out['free_energy']):.4f}  rpe={float(out['rpe']):.4f}")
    # cos(z5) 分化
    keys = list(results.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = torch.nn.functional.cosine_similarity(
                results[keys[i]].unsqueeze(0), results[keys[j]].unsqueeze(0)).item()
            print(f"    cos(Z5) {keys[i][:12]} vs {keys[j][:12]} = {c:.4f}")


def test_memory(net, text=None, repeats=3):
    """世界模型记忆: 重复喂同输入, 自由能是否逐轮下降."""
    if text is None:
        text = "Hello, world! This is a test of memory retention in the world model."
    print(f'\n  [Memory] "{text[:40]}..."')
    dev = next(net.parameters()).device
    bv = _byte_tensor(text, dev)
    fes = []
    for rep in range(repeats):
        out = net(bv)
        fes.append(float(out["free_energy"]))
        print(f"    round {rep + 1}: free_energy={fes[-1]:.4f}")
    if len(fes) > 1 and fes[-1] < fes[0]:
        print(f"    memory: free_energy 下降 {fes[0]:.4f} -> {fes[-1]:.4f} (YES)")
    else:
        print(f"    memory: 未下降 ({fes[0]:.4f} -> {fes[-1]:.4f})")


def test_generation(net, prompts=None, n_tokens=40):
    if prompts is None:
        prompts = ["Hello", "import ", "The quick"]
    print("\n  [Generation]")
    dev = next(net.parameters()).device
    for prompt in prompts:
        out = net.generate(prompt, n_tokens=n_tokens, dev=dev)
        ascii_safe = out[:60].decode("utf-8", errors="replace").encode("ascii", errors="replace").decode("ascii")
        printable = sum(32 <= c < 127 for c in out)
        ratio = printable / max(len(out), 1)
        print(f'    "{prompt}" -> "{ascii_safe}"   ascii_ratio={ratio:.2f}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    for path in args.checkpoints:
        print(f"\n{'=' * 60}")
        print(f"PPA Eval: {path}")
        print(f"{'=' * 60}")
        net, cfg = load_model(path)
        dev = next(net.parameters()).device
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"    device={dev}  trainable_params={trainable:,}")
        test_free_energy(net)
        test_memory(net)
        test_generation(net)


if __name__ == "__main__":
    main()
