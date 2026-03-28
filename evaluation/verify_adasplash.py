"""Verify that adasplash entmax attention is actually being called."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adasplash"))

import torch
from transformers import AutoModelForCausalLM, AutoConfig
import adasplash.adasplash_block_mask as abm

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/adasplash-1b-init"

config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True, attn_implementation="eager").cuda().half()

# Monkey-patch to count calls
_orig_apply = abm._sparse_attention.apply
call_count = [0]

def _tracked_apply(*args, **kwargs):
    call_count[0] += 1
    return _orig_apply(*args, **kwargs)

abm._sparse_attention.apply = _tracked_apply

# Run one forward pass
x = torch.randint(0, 1000, (1, 128)).cuda()
with torch.no_grad():
    out = model(x)

num_layers = config.num_hidden_layers
if call_count[0] == num_layers:
    print(f"OK: adasplash called {call_count[0]} times (1 per layer, {num_layers} layers)")
elif call_count[0] == 0:
    print("FAIL: adasplash was never called — attention is using a different backend")
else:
    print(f"WARNING: adasplash called {call_count[0]} times, expected {num_layers}")
