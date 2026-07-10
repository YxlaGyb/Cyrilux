import sys
print("step1", flush=True)
sys.path.insert(0, r'e:\SystemShare\Documents\virtuosov2')
print("step2", flush=True)
try:
    from dataset.lm_dataset import PretrainDataset
    print("step3 ok", flush=True)
except Exception as e:
    print(f"step3 error: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
print("done", flush=True)
