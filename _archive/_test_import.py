import traceback, sys
sys.path.insert(0, r'e:\SystemShare\Documents\virtuosov2')
try:
    from dataset.lm_dataset import PretrainDataset
    print('ok')
except:
    traceback.print_exc()
