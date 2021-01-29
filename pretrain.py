#!/usr/bin/env python3
"""
Pretraining script.
"""

__author__ = 'Aswin Sivaraman and Minje Kim'
__maintainer__ = 'Aswin Sivaraman'

import argparse
import json
import os
import pathlib
import warnings
from collections import OrderedDict
from typing import Container, Dict, Optional, Set, Union

import cm_constants as C
import cm_data as D
import cm_models as M
import torch
from asteroid.losses.sdr import SingleSrcNegSDR as LossSDR
from pytorch_lightning import seed_everything
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch import Tensor

warnings.filterwarnings('ignore')

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

loss_sisdr = LossSDR('sisdr')
loss_sdsdr = LossSDR('sdsdr')

ROOT = os.path.dirname(os.path.realpath(__file__))


def train_model_contrastive(
  config: dict,
  use_ray: bool = True,
):
  """
  Parameters
  ----------
  config['batch_size']
  config['learning_rate']
  config['model_name']
  config['model_type']
  config['premixture_snr']
  config['pretrain_name']
  config['pretrain_type']
  config['speaker_id']

  Returns
  -------
  {
    state_dict
    num_batches
    vl_loss
    te_result
  }
  """

  print(config)
  raise NotImplementedError

  # fix seed
  seed_everything(0)


def train_model_unimodal(
  config: dict,
  use_ray: bool = True,
):
  """
  Parameters
  ----------
  config['batch_size']
  config['learning_rate']
  config['model_name']
  config['model_type']
  config['premixture_snr']
  config['pretrain_name']
  config['pretrain_type']
  config['speaker_id']

  Returns
  -------
  {
    state_dict
    num_batches
    vl_loss
    te_result
  }
  """

  print(config)

  # fix seed
  seed_everything(0)


  # create local output directory
  output_dir: str = ''
  if config.get('save_to_local', True):
    if config['pretrain_type'] == 1:
      output_dir = os.path.join(ROOT, 'weights_pt', config['model_name'],
                                'multispeaker')
    elif config['pretrain_type'] == 2:
      output_dir = os.path.join(ROOT, 'weights_pt', config['model_name'],
                                'pseudose',
                                'pm{:02d}'.format(config['premixture_snr']),
                                config['speaker_id'])
    pathlib.Path(output_dir).mkdir(0o777, True, True)
    print('[INFO] Will copy results to {}.'.format(output_dir))
    if os.path.exists(os.path.join(output_dir, 'checkpoint')):
      print('[WARN] This will overwrite an existing checkpoint.')
  else:
    print('[INFO] Will not copy best checkpoint / best test results locally.')


  # select device
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda:0'

  # instantiate model
  model: Union[M.NetworkRNN, M.NetworkCTN]
  if config['model_type'] == 0:
    model = M.NetworkRNN(64, 2).to(device)
  elif config['model_type'] == 1:
    model = M.NetworkRNN(128, 2).to(device)
  elif config['model_type'] == 2:
    model = M.NetworkRNN(256, 2).to(device)
  elif config['model_type'] == 3:
    model = M.NetworkCTN().to(device)
  else:
    raise NotImplementedError('Invalid `model_type`, got {}.'.format(
      config['model_type']
    ))

  # instantiate dataloaders
  if config['pretrain_type'] == 1:  # MULTISPEAKER
    train_dataloader = torch.utils.data.DataLoader(
      D.SpeakerAgnosticMixtures(D.speaker_ids_tr), config['batch_size'])
    (vl_x, vl_s, vl_n) = next(iter(torch.utils.data.DataLoader(
      D.SpeakerAgnosticMixtures(D.speaker_ids_vl), 100)))
    vl_x = vl_x.to(device)
    vl_s = vl_s.to(device)
  elif config['pretrain_type'] == 2:  # PSEUDOSE
    train_dataloader = torch.utils.data.DataLoader(
      D.SpeakerSpecificMixtures(
        config['speaker_id'],
        speech_subset='pretrain',
        premixture_snr=config['premixture_snr'],
        noise_subset='free-sound'), config['batch_size'])
    (vl_x, vl_s, vl_n) = next(iter(torch.utils.data.DataLoader(
      D.SpeakerSpecificMixtures(
        config['speaker_id'],
        speech_subset='prevalidation',
        premixture_snr=config['premixture_snr'],
        noise_subset='free-sound'), 100)))
    vl_x = vl_x.to(device)
    vl_s = vl_s.to(device)
  else:
    raise NotImplementedError('Invalid `pretrain_type`, got {}.'.format(
      config['pretrain_type']
    ))

  # instantiate optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

  # setup metrics + state dict
  (num_batches, current_epoch, best_epoch) = (0, 0, 0)
  best_loss: float = 100.
  best_state_dict: Optional[Dict[str, Tensor]] = None

  #
  # begin training loop
  #

  for (tr_x, tr_s, tr_n) in train_dataloader:

    current_epoch += 1
    num_batches += config['batch_size']

    # zero parameter gradients
    optimizer.zero_grad()

    # move data to GPU
    tr_x = tr_x.to(device)
    tr_s = tr_s.to(device)

    # forward propagation
    tr_rs, tr_rn = model(tr_x)

    # calculate loss
    min_len = min(tr_rs.shape[-1], tr_s.shape[-1])
    loss = loss_sdsdr(tr_rs[..., :min_len],
                      tr_s[..., :min_len]).mean()

    # backward propagation + optimize
    loss.backward()
    optimizer.step()

    # only validate every few epochs
    if current_epoch % 10:
      continue

    # validate
    with torch.no_grad():
      vl_rs, vl_rn = model(vl_x)
      min_len = min(vl_rs.shape[-1], vl_s.shape[-1])
      vl_loss = float(loss_sdsdr(vl_rs[..., :min_len],
                     vl_s[..., :min_len]).mean())

    if vl_loss < best_loss:
      best_epoch = current_epoch
      best_loss = vl_loss
      best_state_dict = model.state_dict()

    if use_ray:
      # send intermediate results back to Ray
      tune.report(num_batches=num_batches, vl_loss=vl_loss)
      with tune.checkpoint_dir(best_epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, 'checkpoint')
        torch.save(best_state_dict, path)

    # check for convergence
    if (current_epoch - best_epoch > 1000):
      break

  #
  # end training loop
  #

  # run a test on the best model
  te_result = None
  if config.get('test', True):
    te_result = test_model(
      model_type=config['model_type'],
      state_dict=best_state_dict,
      speaker_ids=config['speaker_id'],
    )

  # save the state dictionary to the local results folders
  if output_dir:
    torch.save(best_state_dict, os.path.join(output_dir, 'checkpoint'))
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as fp:
      json.dump(te_result, fp, indent=2, sort_keys=True)

  # return the state dictionary with validation and test results
  return {
    'state_dict': best_state_dict,
    'num_batches': best_epoch * config['batch_size'],
    'vl_loss': best_loss,
    'te_result': te_result,
  }


def finetune_model(
  config: dict,
  use_ray: bool = True,
):
  """
  Parameters
  ----------
  config['batch_size']
  config['finetune_duration']
  config['learning_rate']
  config['model_name']
  config['model_type']
  config['premixture_snr']
  config['pretrain_type']
  config['speaker_id']

  Returns
  -------
  {
    state_dict
    num_batches
    vl_result
    te_result
  }
  """

  print(config)

  # fix seed
  seed_everything(0)

  # find the local checkpoint
  input_dir: str = ''
  pretrained_weights_file: str = ''
  if config['pretrain_type'] == 0:
    input_dir = os.path.join(ROOT, 'weights_pt', config['model_name'],
                             'random')
  elif config['pretrain_type'] == 1:
    input_dir = os.path.join(ROOT, 'weights_pt', config['model_name'],
                             'multispeaker')
  elif config['pretrain_type'] == 2:
    input_dir = os.path.join(ROOT, 'weights_pt', config['model_name'],
                             'pseudose',
                             'pm{:02d}'.format(config['premixture_snr']),
                             config['speaker_id'])
    pretrained_weights_file = os.path.join(input_dir, 'checkpoint')
    if not os.path.exists(pretrained_weights_file):
      raise ValueError('{} does not exist.'.format(pretrained_weights_file))
  else:
    raise ValueError('No local checkpoint `pretrain_type` provided.')

  print('[INFO] Finetuning weights from {}...'.format(
    pretrained_weights_file
  ))

  # create local output directory
  output_dir: str = ''
  if config.get('save_to_local', True):
    if config['pretrain_type'] == 1:
      output_dir = os.path.join(ROOT, 'weights_ft', config['model_name'],
                                '{:02d}'.format(config['finetune_duration']),
                                'multispeaker',
                                config['speaker_id'])
    elif config['pretrain_type'] == 2:
      output_dir = os.path.join(ROOT, 'weights_ft', config['model_name'],
                                '{:02d}'.format(config['finetune_duration']),
                                'pseudose',
                                'pm{:02d}'.format(config['premixture_snr']),
                                config['speaker_id'])
    pathlib.Path(output_dir).mkdir(0o777, True, True)
    print('[INFO] Will copy results to {}.'.format(output_dir))
    if os.path.exists(os.path.join(output_dir, 'checkpoint')):
      print('[WARN] This will overwrite an existing checkpoint.')
  else:
    print('[INFO] Will not copy best checkpoint / best test results locally.')


  # select device
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda:0'

  # instantiate model
  model: Union[M.NetworkRNN, M.NetworkCTN]
  if config['model_type'] == 0:
    model = M.NetworkRNN(64, 2).to(device)
  elif config['model_type'] == 1:
    model = M.NetworkRNN(128, 2).to(device)
  elif config['model_type'] == 2:
    model = M.NetworkRNN(256, 2).to(device)
  elif config['model_type'] == 3:
    model = M.NetworkCTN().to(device)
  else:
    raise NotImplementedError('Invalid `model_type`, got {}.'.format(
      config['model_type']
    ))

  # load pretrained checkpoint
  model.load_state_dict(torch.load(pretrained_weights_file))

  # instantiate dataloader
  train_dataloader = torch.utils.data.DataLoader(
    D.SpeakerSpecificMixtures(
      config['speaker_id'],
      speech_subset='finetune',
      noise_subset='free-sound',
      dataset_duration=config['finetune_duration']), config['batch_size'])

  # pre-define validation set
  (vl_x, vl_s, vl_n) = next(iter(torch.utils.data.DataLoader(
    D.SpeakerSpecificMixtures(
      config['speaker_id'],
      speech_subset='validation',
      noise_subset='free-sound',
      dataset_duration=config['finetune_duration']), 100)))
  vl_x = vl_x.to(device)
  vl_s = vl_s.to(device)

  # calculate the initial validation set SI-SDR
  # (then keep track during the experiment)
  vl_result = []
  with torch.no_grad():
    vl_rs, vl_rn = model(vl_x)
    min_len = min(vl_rs.shape[-1], vl_s.shape[-1])
    sisdr_in = loss_sisdr(vl_x[..., :min_len], vl_s[..., :min_len])
    sisdr_out = loss_sisdr(vl_rs[..., :min_len], vl_s[..., :min_len])
    vl_result.append(float((sisdr_in - sisdr_out).mean()))

  # instantiate optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

  # setup metrics + state dict
  (num_batches, current_epoch, best_epoch) = (0, 0, 0)
  best_loss: float = 100.
  best_state_dict: Optional[Dict[str, Tensor]] = None

  #
  # begin training loop
  #

  for (tr_x, tr_s, tr_n) in train_dataloader:

    current_epoch += 1
    num_batches += config['batch_size']

    # zero parameter gradients
    optimizer.zero_grad()

    # move data to GPU
    tr_x = tr_x.to(device)
    tr_s = tr_s.to(device)

    # forward propagation
    tr_rs, tr_rn = model(tr_x)

    # calculate loss
    min_len = min(tr_rs.shape[-1], tr_s.shape[-1])
    loss = loss_sdsdr(tr_rs[..., :min_len],
                      tr_s[..., :min_len]).mean()

    # backward propagation + optimize
    loss.backward()
    optimizer.step()

    # only validate every few epochs
    if current_epoch % 10:
      continue

    # validate
    with torch.no_grad():
      vl_rs, vl_rn = model(vl_x)
      min_len = min(vl_rs.shape[-1], vl_s.shape[-1])
      vl_loss = float(loss_sdsdr(vl_rs[..., :min_len],
                                 vl_s[..., :min_len]).mean())
      sisdr_in = loss_sisdr(vl_x[..., :min_len], vl_s[..., :min_len])
      sisdr_out = loss_sisdr(vl_rs[..., :min_len], vl_s[..., :min_len])
      vl_result.append(float((sisdr_in - sisdr_out).mean()))

    if vl_loss < best_loss:
      best_epoch = current_epoch
      best_loss = vl_loss
      best_state_dict = model.state_dict()

    if use_ray:
      # send intermediate results back to Ray
      tune.report(num_batches=num_batches, vl_loss=vl_loss)
      with tune.checkpoint_dir(best_epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, 'checkpoint')
        torch.save(best_state_dict, path)

    # check for convergence
    if (current_epoch - best_epoch > 1000):
      break

  #
  # end training loop
  #

  # run a test on the best model
  te_result = None
  if config.get('test', True):
    te_result = test_model(
      model_type=config['model_type'],
      state_dict=best_state_dict,
      speaker_ids=config['speaker_id'],
    )

  # save the state dictionary to the local results folders
  if output_dir:
    torch.save(best_state_dict, os.path.join(output_dir, 'checkpoint'))
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as fp:
      json.dump(te_result, fp, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, 'validation_sisdr.json'), 'w') as fp:
      json.dump(vl_result[:best_epoch//10], fp, indent=2, sort_keys=True)

  return {
    'state_dict': best_state_dict,
    'num_batches': best_epoch * config['batch_size'],
    'vl_loss': best_loss,
    'vl_result': vl_result[:best_epoch//10],
    'te_result': te_result,
  }


def test_model(
  model_type: int,
  checkpoint_dir: Optional[str] = None,
  state_dict: Optional[Dict[str, Tensor]] = None,
  speaker_ids: Optional[Set[str]] = None,
):
  """Calculate average SI-SDR improvement of a model per speaker.
  """

  # sanity check arguments
  if not state_dict and not checkpoint_dir:
    raise ValueError('Expected either `checkpoint_dir` or `state_dict`.')
  if checkpoint_dir and not os.path.isdir(checkpoint_dir):
    raise ValueError(f'Invalid folder `checkpoint_dir`, got {checkpoint_dir}.')
  if state_dict and not isinstance(state_dict, OrderedDict):
    raise ValueError(f'Expected a valid PyTorch `state_dict`.')
  if speaker_ids is None:
    speaker_ids = D.speaker_ids_te
  if isinstance(speaker_ids, str):
    speaker_ids = [speaker_ids]

  # wrap with the no-gradient context manager
  with torch.no_grad():

    # fix seed
    seed_everything(0)

    # select device
    device = 'cpu'
    if torch.cuda.is_available():
      device = 'cuda:0'

    # instantiate model
    model: Union[M.NetworkRNN, M.NetworkCTN]
    if model_type == 0:
      model = M.NetworkRNN(64, 2).to(device)
    elif model_type == 1:
      model = M.NetworkRNN(128, 2).to(device)
    elif model_type == 2:
      model = M.NetworkRNN(256, 2).to(device)
    elif model_type == 3:
      model = M.NetworkCTN().to(device)

    if checkpoint_dir:
      path = os.path.join(checkpoint_dir, 'checkpoint')
      model.load_state_dict(torch.load(path))
    elif state_dict:
      model.load_state_dict(state_dict)

    te_result = {}

    for speaker_id in speaker_ids:
      # pre-define test set
      (te_x, te_s, te_n) = next(iter(torch.utils.data.DataLoader(
        D.SpeakerSpecificMixtures(
          speaker_id,
          speech_subset='test',
          noise_subset='sound-bible',
          utterance_duration=3), 100)))
      te_x = te_x.to(device)
      te_s = te_s.to(device)

      # run test
      te_rs, te_rn = model(te_x)
      min_len = min(te_rs.shape[-1], te_s.shape[-1])
      sisdr_in = loss_sisdr(te_x[..., :min_len], te_s[..., :min_len])
      sisdr_out = loss_sisdr(te_rs[..., :min_len], te_s[..., :min_len])

      te_result[speaker_id] = str(float((sisdr_in-sisdr_out).mean()))+' dB'

  return te_result


def pretrain_per_speaker_id(
  model_type: int,
  pretrain_type: int,
  premixture_snr: float = 10.0,
  speaker_ids: Optional[Union[str, Container[str]]] = D.speaker_ids_te,
  num_gpus: float = 1.0,
):

  saved_args = locals()

  # convert the int arguments to names
  model_name: str = C.MODEL_NAMES[model_type]
  pretrain_name: str = C.PRETRAIN_NAMES[pretrain_type]

  # select appropriate learning rate based on empirical best-choices
  learning_rate: float = -1.0
  if pretrain_type == 1:
    learning_rate = C.LEARNING_RATES_PT_1[model_name]
  elif pretrain_type == 2:
    learning_rate = C.LEARNING_RATES_PT_2[model_name][premixture_snr]

  # select the appropriate training function
  trial_func = train_model_unimodal
  if pretrain_type == 3:
    trial_func = train_model_contrastive

  # create a configuration dictionary which will be the same across
  # every trial; only the `speaker_id` will vary per run
  config = {
    'model_type': model_type,
    'model_name': model_name,
    'pretrain_type': pretrain_type,
    'pretrain_name': tune.grid_search([pretrain_name]),
    'learning_rate': learning_rate,
    'batch_size': C.BATCH_SIZES[model_name],
    'premixture_snr': (tune.grid_search([premixture_snr])
      if pretrain_type > 1 else None),
    'speaker_id': (tune.grid_search(speaker_ids)
      if pretrain_type > 1 else None),
    'save_to_local': True,
    'test': True,
  }
  print(config)

  # use Tune to queue up trials in parallel on the GPUs
  tune.run(
    trial_func,
    resources_per_trial={'gpu': num_gpus},
    config=config,
    log_to_file='log.txt',
    keep_checkpoints_num=1,
    queue_trials=True,
  )

  print(
    'Finished `pretrain_per_speaker_id(' +
    ', '.join([f'{k}={v}' for (k,v) in saved_args.items()]) + ')`.'
  )


def finetune_per_speaker_id(
  model_type: int,
  pretrain_type: int,
  premixture_snr: float = 10.0,
  speaker_ids: Optional[Union[str, Container[str]]] = D.speaker_ids_te,
  finetune_duration: int = 3,
  num_gpus: float = 1.0,
):

  saved_args = locals()

  # convert the int arguments to names
  model_name: str = C.MODEL_NAMES[model_type]
  pretrain_name: str = C.PRETRAIN_NAMES[pretrain_type]

  # select appropriate learning rate based on empirical best-choices
  learning_rate: float = C.LEARNING_RATES_FT[model_name][finetune_duration]

  # create a configuration dictionary which will be the same across
  # every trial; only the `speaker_id` will vary per run
  config = {
    'finetune_duration': finetune_duration,
    'model_type': model_type,
    'model_name': model_name,
    'pretrain_type': pretrain_type,
    'pretrain_name': tune.grid_search([pretrain_name]),
    'learning_rate': learning_rate,
    'batch_size': C.BATCH_SIZES[model_name],
    'premixture_snr': (tune.grid_search([premixture_snr])
      if pretrain_type > 1 else None),
    'speaker_id': (tune.grid_search(speaker_ids)
      if pretrain_type > 1 else None),
    'save_to_local': True,
    'test': True,
  }
  print(config)

  # use Tune to queue up trials in parallel on the GPUs
  tune.run(
    finetune_model,
    resources_per_trial={'gpu': num_gpus},
    config=config,
    log_to_file='log.txt',
    keep_checkpoints_num=1,
    queue_trials=True,
  )

  print(
    'Finished `finetune_per_speaker_id(' +
    ', '.join([f'{k}={v}' for (k,v) in saved_args.items()]) + ')`.'
  )


def pretrain_find_learning_rate(
  model_type: int,
  pretrain_type: int,
  premixture_snr: float = 10.0,
  num_gpus: float = 1.0,
  num_samples: int = 20,
):

  saved_args = locals()

  # convert the int arguments to names
  model_name: str = C.MODEL_NAMES[model_type]
  pretrain_name: str = C.PRETRAIN_NAMES[pretrain_type]

  # select the appropriate training function
  trial_func = train_model_unimodal
  if pretrain_type == 3:
    trial_func = train_model_contrastive

  # create a configuration dictionary which will be the same across
  # every trial; only the `speaker_id` will vary per run
  config = {
    'model_type': model_type,
    'model_name': model_name,
    'pretrain_type': pretrain_type,
    'pretrain_name': tune.grid_search([pretrain_name]),
    'learning_rate': tune.loguniform(1e-7, 1e-2),
    'batch_size': C.BATCH_SIZES[model_name],
    'premixture_snr': (tune.grid_search([premixture_snr])
      if pretrain_type > 1 else None),
    'speaker_id': (tune.grid_search(['200'])
      if pretrain_type > 1 else None),
    'save_to_local': True,
    'test': True,
  }
  print(config)

  scheduler = ASHAScheduler(
    metric='vl_loss',
    mode='min',
    max_t=10000,
    grace_period=500,
    reduction_factor=2
  )

  # use Tune to queue up trials in parallel on the GPUs
  result = tune.run(
    trial_func,
    resources_per_trial={'gpu': num_gpus},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    log_to_file='log.txt',
    keep_checkpoints_num=1,
    queue_trials=True,
  )

  print('Best trial config: {}'.format(
    best_trial.config))

  print('Best trial learning_rate: {}'.format(
    best_trial.config['learning_rate']))

  print('Best trial final validation loss: {}'.format(
    best_trial.last_result['vl_loss']))

  print('Best model weights saved at: {}'.format(
    best_trial.checkpoint.value))

  with open(str(best_trial.checkpoint.value)+'/best_lr.txt', 'w') as fp:
    json.dump(best_trial.config['learning_rate'], fp)
    print('Wrote result to {}.'.format(
      str(best_trial.checkpoint.value)+'/best_lr.txt'
    ))

  print(
    'Finished `pretrain_find_learning_rate(' +
    ', '.join([f'{k}={v}' for (k,v) in saved_args.items()]) + ')`.'
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument(
      '-i', '--pretrain_type',
      required=True,
      type=int, choices={0, 1, 2, 3},
      help=('method of network initialization: \n'
            '\u2022 0: Gaussian random,\n'
            '\u2022 1: transfer-learning from multispeaker,\n'
            '\u2022 2: unimodal remixtures,\n'
            '\u2022 3: contrastive remixtures')
  )
  parser.add_argument(
      '-n', '--architecture_type',
      required=True,
      type=int, choices={0, 1, 2, 3},
      help=('supported network architectures: \n'
            '\u2022 0: GRU(64 hidden_units, 2 layers) + Dense,\n'
            '\u2022 1: GRU(128 hidden_units, 2 layers) + Dense,\n'
            '\u2022 2: GRU(256 hidden_units, 2 layers) + Dense,\n'
            '\u2022 3: ConvTasNet')
  )
  parser.add_argument(
      '-g', '--num_gpus',
      required=True,
      type=float,
      help=('number of GPUs to use for each trial (fractional values\n'
            'can be used)')
  )
  parser.add_argument(
      '-p', '--premixture_snr',
      required=True,
      type=int, choices={0, 5, 10},
      help=('premixture SNR (in dB)')
  )
  parser.add_argument(
      '-d', '--finetune_duration', metavar='DURATION',
      type=int, default=0,
      help=('amount of clean speech speaker-specific data \n'
            'specified in seconds (default: 0)')
  )
  parser.add_argument(
      '-s', '--speaker_ids', type=str, nargs='+', default=D.speaker_ids_te,
      help='optional flag for specific speaker IDs'
  )
  args = parser.parse_args()
  print(args)
  if args.finetune_duration > 0:
    finetune_per_speaker_id(model_type=args.architecture_type,
                            pretrain_type=args.pretrain_type,
                            premixture_snr=args.premixture_snr,
                            finetune_duration=args.finetune_duration,
                            speaker_ids=args.speaker_ids,
                            num_gpus=args.num_gpus)
  else:
    pretrain_per_speaker_id(model_type=args.architecture_type,
                            pretrain_type=args.pretrain_type,
                            premixture_snr=args.premixture_snr,
                            speaker_ids=args.speaker_ids,
                            num_gpus=args.num_gpus)
