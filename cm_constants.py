from typing import Dict

MODEL_NAMES: Dict[int, str] = {
  0: 'NetworkRNN_0064x02',
  1: 'NetworkRNN_0128x02',
  2: 'NetworkRNN_0256x02',
  3: 'NetworkCTN',
}

PRETRAIN_NAMES: Dict[int, str] = {
  0: 'random',
  1: 'multispeaker',
  2: 'pseudose',
  3: 'contrastive',
}

# usage: BATCH_SIZES[model_name]
#
BATCH_SIZES: Dict[str, int] = {
  'NetworkRNN_0064x02': 128,
  'NetworkRNN_0128x02': 128,
  'NetworkRNN_0256x02': 128,
  'NetworkCTN': 48,
}

# empirical rounded-up model space allocation in mebibytes (MiB)
# usage: GPU_ALLOCATION[model_name]
#
GPU_ALLOCATION: Dict[str, float] = {
  'NetworkRNN_0064x02': 1400,
  'NetworkRNN_0128x02': 1450,
  'NetworkRNN_0256x02': 1500,
  'NetworkCTN': 9600,
}

# usage: LEARNING_RATES_PT_1[model_name]
#
LEARNING_RATES_PT_1: Dict[str, float] = {
  'NetworkRNN_0064x02': 5e-4,
  'NetworkRNN_0128x02': 7e-4,
  'NetworkRNN_0256x02': 7e-4,
  'NetworkCTN': 1e-3,
}

# usage: LEARNING_RATES_PT_2[model_name][premixture_snr]
#
LEARNING_RATES_PT_2: Dict[str, Dict[float, float]] = {
  'NetworkRNN_0064x02': {0: 2e-3, 5: 1e-3, 10: 2e-3},
  'NetworkRNN_0128x02': {0: 2e-3, 5: 5e-4, 10: 4e-4},
  'NetworkRNN_0256x02': {0: 8e-4, 5: 6e-4, 10: 9e-4},
  'NetworkCTN': {0: 1e-3, 5: 7e-4, 10: 8e-4},
}

# e.g.
# learning_rates_ft[model_name][finetune_duration]
LEARNING_RATES_FT: Dict[str, Dict[int, float]] = {
  'NetworkRNN_0064x02': {3: 1e-6, 5: 1e-6, 10: 5e-6, 30: 7e-6, 60: 3e-5},
  'NetworkRNN_0128x02': {3: 3e-7, 5: 9e-7, 10: 8e-7, 30: 5e-6, 60: 4e-6},
  'NetworkRNN_0256x02': {3: 2e-7, 5: 3e-7, 10: 6e-7, 30: 2e-5, 60: 2e-5},
  'NetworkCTN':         {3: 2e-6, 5: 2e-6, 10: 3e-6, 30: 8e-6, 60: 1e-5},
}
