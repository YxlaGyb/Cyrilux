"""端到端性能基准: warmup -> connect_layer -> Hebbian training."""
import time
import torch
from model.model_cyrene import CyreneConfig, CyreneModel

# 接近生产环境的配置
cfg = CyreneConfig(
    hidden_size=64,
    warmup_steps=20,
    max_neurons=65536,
    max_synapses=8_000_000,
    hidden_neurons=256,
)
m = CyreneModel(cfg)
m.add_hidden_layer(n_neurons=256)
dummy = torch.zeros(1, 2, 64, dtype=torch.half, device=m.device)

# Warmup phase
t0 = time.perf_counter()
warmup_steps = cfg.warmup_steps
for s in range(warmup_steps):
    m.step(dummy)
warmup_elapsed = time.perf_counter() - t0
print(f"Warmup ({warmup_steps} steps): {warmup_elapsed:.2f}s ({warmup_steps/warmup_elapsed:.1f} it/s)")

# Transition step (step 51: connect_layer + first Hebbian)
t0 = time.perf_counter()
m.step(dummy)
transition_elapsed = time.perf_counter() - t0
print(f"Transition step (connect+Hebbian): {transition_elapsed:.2f}s")

# Post-warmup training
n_train = 50
t0 = time.perf_counter()
for s in range(n_train):
    m.step(dummy, target_byte=ord('a'))
train_elapsed = time.perf_counter() - t0
print(f"Training ({n_train} steps): {train_elapsed:.2f}s ({n_train/train_elapsed:.1f} it/s)")

# Detailed phase breakdown post-warmup
print("\nPost-warmup per-step phases (5 samples):")
for _ in range(5):
    m._step += 1
    t = {}

    t0 = time.perf_counter()
    h_list = m.encode(dummy)
    t1 = time.perf_counter()
    events = m.ingest(h_list, top_k=4)
    t2 = time.perf_counter()
    m.process_sensory_events(events)
    t3 = time.perf_counter()
    m.predict_pass()
    t4 = time.perf_counter()
    free_energy, _, _ = m.compute_stats()
    t5 = time.perf_counter()
    m.pool.emit_active(m._step)
    t6 = time.perf_counter()
    m.pool.adjust_thresholds()
    t7 = time.perf_counter()
    m.modulate(free_energy)
    t8 = time.perf_counter()
    m.hebbian_pass(m._last_modulation)
    t9 = time.perf_counter()

    print(f"  encode={1000*(t1-t0):.1f}ms ingest={1000*(t2-t1):.1f}ms "
          f"sensory={1000*(t3-t2):.1f}ms predict={1000*(t4-t3):.1f}ms "
          f"stats={1000*(t5-t4):.1f}ms emit+adj={1000*(t7-t6):.1f}ms "
          f"modulate={1000*(t8-t7):.1f}ms hebbian={1000*(t9-t8):.1f}ms "
          f"total={1000*(t9-t0):.1f}ms")

activity = m.pool.get_activity_stats()
print(f"\nFinal: {activity['total_neurons']} neurons, {activity['total_synapses']} synapses")
