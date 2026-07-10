import sys, traceback
sys.path.insert(0, 'e:\\SystemShare\\Documents\\virtuosov2')
out = open('e:\\SystemShare\\Documents\\virtuosov2\\_trace.log', 'w', encoding='utf-8')
try:
    out.write('1:torch\n'); out.flush(); import torch
    out.write('2:model\n'); out.flush(); from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    out.write('3:datasets\n'); out.flush(); from datasets import load_dataset
    out.write('4:transformers\n'); out.flush(); from transformers import AutoTokenizer
    out.write('5:tqdm\n'); out.flush(); from tqdm import tqdm
    out.write('6:DONE\n')
except:
    out.write('ERROR: ' + traceback.format_exc() + '\n')
finally:
    out.close()
