"""
训练管理器: 桥接 GUI 与 train_pc_unified.py，在后台线程中管理训练流程。

职责:
  - 接收 GUI 配置 → 构建训练参数
  - 在后台线程启动训练循环
  - 通过回调/队列向 GUI 推送实时进度
  - 管理检查点加载/保存
  - 训练结束后可无缝过渡到 Phase 2 (autonomous_mind)
"""
import os, sys, json, math, threading, queue, time, traceback
from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# 将项目根目录加入 sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig
from model.pc_core import DopamineSignal
from trainer_utils import get_lr, Logger, setup_seed
from data_converter import convert_sample


# ── 进度回调类型 ──
ProgressCallback = Callable[[dict], None]
"""回调参数 dict:
    {
        'type': 'log' | 'progress' | 'phase' | 'checkpoint' | 'done' | 'error',
        'message': str,
        'step': int,
        'total_steps': int,
        'ce_loss': float,
        'F': float,
        'D': float,
        'lr': float,
        'epoch': int,
        'phase_name': str,
        'checkpoint_path': str,
    }
"""


@dataclass
class TrainingConfig:
    """训练配置 — 从 GUI 收集后填充。"""
    # 模型选择
    model_type: str = 'new'            # 'new' | 'checkpoint'
    checkpoint_path: str = ''          # 当 model_type='checkpoint' 时使用
    hidden_size: int = 256
    num_hidden_layers: int = 4

    # 数据选择
    data_files: list = field(default_factory=list)  # 数据文件路径列表
    combined_training: bool = True     # True=多文件一起训练, False=逐个训练

    # 训练参数
    batch_size: int = 48
    max_seq_len: int = 128
    lr: float = 3e-4
    epochs: int = 1
    subset: int = 0                    # 0 = 使用全部

    # PC 参数
    T_infer: int = 1
    gamma: float = 0.1

    # 多巴胺
    enable_dopamine: bool = True
    dopamine_eta: float = 1.0
    dopamine_beta: float = 0.5
    dopamine_gamma: float = 0.3

    # QAT
    enable_quantize: bool = False

    # 输出
    out_dir: str = 'out_pc_unified'
    save_interval: int = 500

    # 持续学习 (Phase 1 暂不使用全套持续学习, 但保留)
    use_abstraction_bank: bool = False

    # 训练结束后自动进入 Phase 2
    auto_start_phase2: bool = False

    def to_cli_args(self) -> list:
        """转换为命令行参数列表 (供 subprocess 或重构使用)。"""
        args = [
            '--batch_size', str(self.batch_size),
            '--max_seq_len', str(self.max_seq_len),
            '--lr', str(self.lr),
            '--epochs', str(self.epochs),
            '--T_infer', str(self.T_infer),
            '--gamma', str(self.gamma),
            '--out_dir', self.out_dir,
            '--save_interval', str(self.save_interval),
        ]
        if self.subset > 0:
            args.extend(['--subset', str(self.subset)])
        if self.enable_dopamine:
            args.extend(['--dopamine',
                         '--dopamine_eta', str(self.dopamine_eta),
                         '--dopamine_beta', str(self.dopamine_beta),
                         '--dopamine_gamma', str(self.dopamine_gamma)])
        if self.enable_quantize:
            args.append('--quantize')
        return args


# ═══════════════════════════════════════════════════════════════
# _LocalDataset — 支持多文件合并加载
# ═══════════════════════════════════════════════════════════════

class _LocalDataset(torch.utils.data.Dataset):
    """原始 UTF-8 字节数据集，支持多文件合并。"""
    def __init__(self, data_paths: list, max_length=128, max_samples=None):
        super().__init__()
        self.byte_tensors = []
        self.label_tensors = []
        self._file_offsets = {}  # 文件 → (start_idx, end_idx)

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        for fpath in data_paths:
            start_idx = len(self.byte_tensors)
            with open(fpath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and len(self.byte_tensors) >= max_samples:
                        break
                    sample = json.loads(line)
                    # 支持 text 和 conversations 两种格式
                    if 'text' in sample:
                        text = sample['text']
                    elif 'conversations' in sample:
                        from data_converter import conversations_to_text
                        text = conversations_to_text(sample['conversations'], sample.get('gt', None))
                    else:
                        text = str(sample)

                    byte_seq = text.encode('utf-8')[:max_length]
                    padded = byte_seq.ljust(max_length, b'\x00')
                    t = torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone()
                    self.byte_tensors.append(t)
                    lbl = t.clone()
                    lbl[t == 0x00] = -100
                    self.label_tensors.append(lbl.to(torch.long))

            end_idx = len(self.byte_tensors)
            self._file_offsets[os.path.basename(fpath)] = (start_idx, end_idx)

    def __len__(self):
        return len(self.byte_tensors)

    def __getitem__(self, index):
        return self.byte_tensors[index].clone(), self.label_tensors[index].clone()


# ═══════════════════════════════════════════════════════════════
# TrainManager
# ═══════════════════════════════════════════════════════════════

class TrainManager:
    """后台训练管理器。在独立线程中运行训练循环，通过回调推送到 GUI。"""

    def __init__(self, config: TrainingConfig, progress_callback: ProgressCallback = None):
        self.config = config
        self.callback = progress_callback or (lambda x: None)
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._trained_model = None  # 训练后的模型，供 Phase 2 使用
        self._lm_config = None
        self._final_state = {}      # 保存最终状态

    # ── 公开接口 ──

    def start(self):
        """启动后台训练。非阻塞。"""
        if self._thread and self._thread.is_alive():
            self._log('训练已在运行中')
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()
        self._log('训练线程已启动')

    def stop(self):
        """请求停止训练。"""
        self._stop_flag.set()
        self._log('正在停止训练...')

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_model(self):
        """获取训练后的模型 (供 Phase 2 使用)。"""
        return self._trained_model

    def get_final_state(self) -> dict:
        return self._final_state

    def wait(self, timeout=None):
        """等待训练完成。"""
        if self._thread:
            self._thread.join(timeout)

    # ── 内部训练循环 ──

    def _log(self, msg: str):
        self.callback({'type': 'log', 'message': msg})

    def _progress(self, **kwargs):
        kwargs['type'] = 'progress'
        self.callback(kwargs)

    def _emit(self, **kwargs):
        self.callback(kwargs)

    def _build_data_paths(self) -> list:
        """根据配置确定实际训练数据文件列表。"""
        data_files = self.config.data_files
        if not data_files:
            default = os.path.join(ROOT, 'datasets', 'pretrain_t2t_mini.jsonl')
            if os.path.exists(default):
                data_files = [default]
            else:
                raise FileNotFoundError('没有选择数据文件，也未找到默认数据集')
        return data_files

    def _run_training(self):
        try:
            setup_seed(42)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self._log(f'设备: {device}')

            # ── 模型配置 ──
            self._lm_config = MiniMindConfig(
                hidden_size=self.config.hidden_size,
                num_hidden_layers=self.config.num_hidden_layers,
                use_moe=False,
            )

            # ── 模型创建 ──
            if self.config.model_type == 'checkpoint' and self.config.checkpoint_path:
                self._log(f'从检查点加载: {self.config.checkpoint_path}')
                ckpt = torch.load(self.config.checkpoint_path, map_location='cpu', weights_only=False)
                self._lm_config = ckpt.get('lm_config', self._lm_config)
                pc_model = PCLocalDynamicMiniMind(self._lm_config)
                pc_model.load_state_dict(ckpt['model_state'], strict=False)
                self._log(f'模型加载完成, 参数量: {sum(p.numel() for p in pc_model.parameters())/1e6:.2f}M')
            else:
                self._log('创建新模型...')
                pc_model = PCLocalDynamicMiniMind(self._lm_config)
                base_params = sum(p.numel() for p in pc_model.parameters() if p.requires_grad)
                self._log(f'新模型创建完成, 参数量: {base_params/1e6:.2f}M')

            # ── QAT ──
            quantizer = None
            if self.config.enable_quantize:
                from torchao.quantization.qat import Int4WeightOnlyQATQuantizer
                quantizer = Int4WeightOnlyQATQuantizer(groupsize=64, inner_k_tiles=4)
                pc_model = quantizer.prepare(pc_model)
                self._log('4bit QAT 已启用')

            pc_model = pc_model.to(device)
            if device == 'cuda:0':
                torch.set_float32_matmul_precision('medium')
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True

            # ── 数据 ──
            data_paths = self._build_data_paths()
            self._log(f'数据文件: {len(data_paths)} 个')
            for p in data_paths:
                self._log(f'  {os.path.basename(p)}')

            ds = _LocalDataset(data_paths,
                               max_length=self.config.max_seq_len,
                               max_samples=self.config.subset if self.config.subset > 0 else None)
            loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True,
                                num_workers=0, pin_memory=True)
            iters = len(loader)
            total_samples = len(ds)
            self._log(f'数据加载完成: {total_samples} 样本, {iters} 步/epoch')

            # ── 预热 ──
            with torch.no_grad():
                dummy = torch.randint(0, 256, (self.config.batch_size, self.config.max_seq_len), device=device)
                dummy_pos = pc_model.get_position_embeddings(self.config.max_seq_len, device)
                _, _ = pc_model.forward_with_ce(dummy, dummy, dummy_pos)
            self._log('预热完成')

            # ── 优化器 ──
            optimizer = torch.optim.AdamW(
                list(pc_model.temporal_proj.parameters()) +
                list(pc_model.topdown_proj.parameters()) +
                [p for n, p in pc_model.model.named_parameters() if p.requires_grad],
                lr=self.config.lr, betas=(0.9, 0.95), fused=device.startswith('cuda'),
            )

            out_dir = os.path.join(ROOT, self.config.out_dir)
            os.makedirs(out_dir, exist_ok=True)

            # ── 多巴胺 ──
            dopamine = None
            if self.config.enable_dopamine:
                dopamine = DopamineSignal(η=self.config.dopamine_eta, threshold=0.0)
                dopamine.F_prev = None
                _dopamine_warmup_steps = 10

            # ── 训练循环 ──
            pc_model.train()
            global_step = 0
            total_steps = iters * self.config.epochs
            prev_precision_scales = None
            ema_z = None
            D = 0.5

            self._emit(type='phase', phase_name='train_loop',
                       message=f'开始训练: {total_steps} 步')

            for epoch in range(self.config.epochs):
                if self._stop_flag.is_set():
                    self._log('训练被用户中断')
                    break

                for step, (byte_seq, labels) in enumerate(loader):
                    if self._stop_flag.is_set():
                        break

                    byte_seq = byte_seq.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    bsz, seq_len = byte_seq.shape
                    global_step += 1

                    # Phase 1: 共享前向
                    pos_emb = pc_model.get_position_embeddings(seq_len, device)
                    z_init, ce_loss = pc_model.forward_with_ce(byte_seq, labels, pos_emb)

                    # Phase 2+: PC 推理
                    z_detached = [z.detach() for z in z_init]
                    z_converged, errors_hist, F_hist, F_pred = pc_model.spatiotemporal_infer(
                        z_detached, pos_emb, gamma=self.config.gamma, T=self.config.T_infer,
                        return_errors=self.config.enable_dopamine,
                        return_pred_loss=True,
                        precision_scales=prev_precision_scales,
                    )

                    # Phase 3: 损失合并
                    ce_converged = ce_loss
                    β_local = min(2.0, 0.1 + global_step / max(total_steps, 1) * 1.9)
                    β_conv = min(1.0, 0.0 + global_step / max(total_steps, 1) * 1.0)

                    # Phase 4: 多巴胺
                    if self.config.enable_dopamine and dopamine is not None:
                        if dopamine.F_prev is None:
                            dopamine.F_prev = float('inf')
                            D = 0.5 if global_step <= _dopamine_warmup_steps else 0.0
                        else:
                            D = dopamine.update(F_pred.item())

                    β_local = β_local * (1.0 + self.config.dopamine_gamma * D)
                    β_conv = β_conv * (1.0 + self.config.dopamine_gamma * D)

                    last_errors = errors_hist[-1] if errors_hist else []
                    if last_errors:
                        err_norms = torch.tensor([e[1] for e in last_errors], device=device)
                        max_err = err_norms.max() + 1e-8
                        π_list = 1.0 + self.config.dopamine_eta * D * (err_norms / max_err)
                        prev_precision_scales = π_list.detach().cpu().tolist()
                    else:
                        prev_precision_scales = None

                    # 合并损失
                    ce_local_sum = ce_loss * (bsz * seq_len)
                    ce_conv_sum = ce_converged * (bsz * seq_len)
                    scale_local = (F_pred.detach() / (ce_local_sum.detach() + 1e-8)).clamp(0.1, 10.0)
                    scale_conv = (F_pred.detach() / (ce_conv_sum.detach() + 1e-8)).clamp(0.1, 10.0)
                    total_loss = F_pred + β_local * scale_local * ce_local_sum \
                                          + β_conv * scale_conv * ce_conv_sum

                    # Phase 5: 反向传播
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()

                    trainable_params = [p for p in pc_model.parameters()
                                       if p.requires_grad and p.grad is not None]
                    if trainable_params:
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

                    current_lr = get_lr(global_step, total_steps, self.config.lr)
                    if self.config.enable_dopamine:
                        current_lr = current_lr * (1.0 + self.config.dopamine_beta * D)
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr

                    optimizer.step()

                    # ── 进度推送 ──
                    ce_val = ce_loss.item()
                    ce_conv_val = ce_converged.item() if hasattr(ce_converged, 'item') else ce_val
                    F_val = F_pred.item()
                    F_final = F_hist[-1] if F_hist else 0.0

                    self._progress(
                        step=global_step,
                        total_steps=total_steps,
                        epoch=epoch + 1,
                        total_epochs=self.config.epochs,
                        ce_loss=ce_val,
                        F=F_final,
                        D=D,
                        lr=current_lr,
                    )

                    # ── 详细日志 (每 50 步) ──
                    if (step + 1) % 50 == 0:
                        log_msg = (
                            f'Epoch {epoch+1}/{self.config.epochs} '
                            f'Step {step+1}/{iters} '
                            f'CE={ce_val:.4f} F={F_final:.1f} '
                            f'D={D:.3f} lr={current_lr:.2e}'
                        )
                        self._log(log_msg)

                    # ── Checkpoint ──
                    if (step + 1) % self.config.save_interval == 0 or global_step == 1:
                        ckpt_path = os.path.join(out_dir, f'unified_ckpt_s{step}.pt')
                        ckpt = {
                            'epoch': epoch,
                            'step': step,
                            'global_step': global_step,
                            'model_state': pc_model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'F': F_final,
                            'CE_local': ce_val,
                            'D': D,
                            'lm_config': self._lm_config,
                            'config': self.config,
                        }
                        torch.save(ckpt, ckpt_path)
                        self._emit(type='checkpoint', checkpoint_path=ckpt_path,
                                   message=f'检查点已保存: step {step}')

                # epoch 结束
                self._log(f'Epoch {epoch+1}/{self.config.epochs} 完成')

            # ── 训练结束 ──
            self._trained_model = pc_model.cpu()
            self._final_state = {
                'model': pc_model,
                'lm_config': self._lm_config,
                'optimizer_state': optimizer.state_dict(),
                'final_ce': ce_val,
                'final_F': F_final,
                'final_D': D,
                'total_steps': global_step,
            }

            # 保存最终模型
            final_path = os.path.join(out_dir, 'unified_final.pt')
            final_ckpt = {
                'model_state': pc_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'F': F_final,
                'CE_local': ce_val,
                'D': D,
                'lm_config': self._lm_config,
                'config': self.config,
            }
            torch.save(final_ckpt, final_path)
            self._log(f'最终模型已保存: {final_path}')

            # ── QAT 转换 ──
            if self.config.enable_quantize and quantizer is not None:
                self._log('转换 int4 推理格式...')
                try:
                    pc_model_cpu = pc_model.cpu()
                    pc_model_cpu = quantizer.convert(pc_model_cpu)
                    torch.save(pc_model_cpu.state_dict(), os.path.join(out_dir, 'int4_model.pt'))
                    self._log('Int4 模型已保存')
                except Exception as e:
                    self._log(f'Int4 转换失败 (不影响主模型): {e}')

            self._emit(type='done', message='训练完成！',
                       model=pc_model, config=self._lm_config)

        except Exception as e:
            tb = traceback.format_exc()
            self._emit(type='error', message=f'训练出错: {str(e)}\n{tb}')
            self._log(f'错误: {str(e)}')


# ── 单次独立运行入口 ──

def run_training_standalone(config: TrainingConfig):
    """同步运行训练 (非 GUI 模式)。"""
    mgr = TrainManager(config, progress_callback=lambda x: print(f'[{x.get("type","?")}] {x.get("message","")}'))
    mgr.start()
    mgr.wait()
    return mgr.get_model()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TrainManager 独立运行')
    parser.add_argument('--data_files', type=str, nargs='+', default=[],
                        help='训练数据文件')
    parser.add_argument('--model_type', type=str, default='new',
                        choices=['new', 'checkpoint'])
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--out_dir', type=str, default='out_pc_unified')
    args = parser.parse_args()

    config = TrainingConfig(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        data_files=args.data_files,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
    )
    run_training_standalone(config)
