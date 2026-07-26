import torch, time
from model.model_cyrene import CyreneConfig, CyreneModel

cfg = CyreneConfig(hidden_size=64, warmup_steps=10, max_neurons=16384, max_synapses=200000, hidden_neurons=64)
m = CyreneModel(cfg)
m.add_hidden_layer(n_neurons=64)
dummy = torch.zeros(1, 2, 64, dtype=torch.half, device=m.device)

for _ in range(5):
    m.step(dummy)

# Detailed stage profiling
times = {k: [] for k in ['encode','ingest','xfer','match','create_batch','update_batch','predict','stats','emit']}

for _ in range(20):
    m._step += 1
    is_w = m._step <= m.config.warmup_steps
    
    t0 = time.perf_counter()
    h_list = m.encode(dummy)
    t1 = time.perf_counter()
    events = m.ingest(h_list, top_k=(0 if is_w else 4))
    t2 = time.perf_counter()
    
    if events and len(events) > 0:
        E = min(500, len(events))
        layers_l = events.layer[:E].cpu().tolist()
        positions_l = events.pos[:E].cpu().tolist()
        channels_l = events.ch[:E].cpu().tolist()
        vals_l = events.val[:E].cpu().tolist()
        t2b = time.perf_counter()
        
        matched_nids, matched_ev, unmatched_ev = m.pool.match_sensory_events(layers_l, positions_l, channels_l)
        t2c = time.perf_counter()
        
        if unmatched_ev:
            new_nids = m.pool.create_neurons_batch(
                layers=[layers_l[e] for e in unmatched_ev],
                positions=[positions_l[e] for e in unmatched_ev],
                channels=[channels_l[e] for e in unmatched_ev],
                thresholds=[0.05] * len(unmatched_ev),
            )
        else:
            new_nids = []
        t2d = time.perf_counter()
        
        all_nids_list = matched_nids + new_nids
        all_z = [vals_l[e] for e in matched_ev] + [vals_l[e] for e in unmatched_ev]
        all_nids = torch.tensor(all_nids_list, dtype=torch.int32, device=m.device)
        z_new = torch.tensor(all_z, dtype=torch.float16, device=m.device)
        m.pool.update_batch(all_nids, z_new)
        t2e = time.perf_counter()
    else:
        t2b = t2c = t2d = t2e = t2
    
    m.process_network_events(max_events=10)
    t3 = time.perf_counter()
    m.predict_pass()
    t4 = time.perf_counter()
    m.compute_stats()
    t5 = time.perf_counter()
    m.pool.emit_active(m._step)
    t6 = time.perf_counter()
    
    if events:
        times['encode'].append((t1-t0)*1000)
        times['ingest'].append((t2-t1)*1000)
        times['xfer'].append((t2b-t2)*1000)
        times['match'].append((t2c-t2b)*1000)
        times['create_batch'].append((t2d-t2c)*1000)
        times['update_batch'].append((t2e-t2d)*1000)
        times['predict'].append((t4-t3)*1000)
        times['stats'].append((t5-t4)*1000)
        times['emit'].append((t6-t5)*1000)

print('Stage breakdown (avg ms) after batch fix:')
for k in ['encode','ingest','xfer','match','create_batch','update_batch','predict','stats','emit']:
    if times[k]:
        avg = sum(times[k]) / len(times[k])
        print(f'  {k:13s}: {avg:7.2f} ms')
total = sum(sum(v)/len(v) for v in times.values() if v)
print(f'  {\"TOTAL\":13s}: {total:7.2f} ms')