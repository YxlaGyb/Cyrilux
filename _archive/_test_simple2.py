# test
import sys, traceback
sys.path.insert(0, 'e:\\SystemShare\\Documents\\virtuosov2')
try:
    import dataset.lm_dataset
    print('ok')
except:
    traceback.print_exc()
print('done')
