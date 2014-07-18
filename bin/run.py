#!/usr/bin/env python

import argparse
import copy
import os
import sys

import pfacets

########### SET UP PATH / PARSE COMMAND-LINE ARGUMENTS

# TODO fix this-- the issue is that tokyo is not on the load path
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__),  '..')))
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'sparco', 'qn')))
for x in ['pfacets', 'traceutil']:
  sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..',
    '..', 'lib', x)))

import sparco
import sparco.trace

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--channels',
    help='comma-separated string of channel ranges and individual values')
arg_parser.add_argument('-C', '--local-config-path',
    help='path to a python module containing a configuration dictionary')
arg_parser.add_argument('-i', '--input-path',
    help='path to directory containing input files')
arg_parser.add_argument('-l', '--log-level',
    help='level of logging: DEBUG, INFO, WARNING, ERROR, CRITICAL')
arg_parser.add_argument('-L', '--log-path',
    help='path to log file')
arg_parser.add_argument('--no-output', action='store_false', dest='produce_output',
    help='do not write any output')
arg_parser.add_argument('-O', '--produce-output', action='store_true',
    help='enables output of logging and basis snapshots; not needed if output_path provided')
arg_parser.add_argument('-o', '--output-root',
    help='path to directory containing output files')
arg_parser.add_argument('-s', '--snapshot-interval', default=100, type=int,
    help='number of iterations between basis snapshots')
arg_parser.add_argument('--trial-directory',
    help='name of trial directory located inside root')
arg_parser.set_defaults(
    inner_directory=True,
    produce_output=True
    )

def expand_channel_spec(channel_spec):
  parts = channel_spec.split(',')
  channels = []
  for p in parts:
    if re.search('-', p):
      start, end = p.split('-')
      channels += range(int(start), int(end)+1)
    else:
      channels.append(int(p))
  return channels

def parse_args():
  global arg_parser
  args = arg_parser.parse_args()
  if args.channels:
    args.channels = expand_channel_spec(args.channels)
  return args

########### BUILD CONFIGURATON

args = parse_args()
cli_config = pfacets.map_object_to_dict(args, {
    'input_path': ['sampler', 'input_path'],
    'output_root': ['trace', 'output_root'],
    'snapshot_interval': ['trace', 'snapshot_interval'],
    'produce_output': ['produce_output'],
    'trial_directory': ['trace', 'trial_directory']
    })
config_module = pfacets.load_local_module(path = args.local_config_path,
    default_name='.sparcorc')
local_config = config_module.config if config_module else {}
config = pfacets.merge(local_config, cli_config)
config.setdefault('run_ladder', True)

########### RUN

def run_coder(config):
  sc = sparco.sparse_coder.SparseCoder(config['nets'])
  if config['produce_output'] or config['output_root']:
    sparco.trace.configure(coder=sc, **config['trace'])
  sc.run()

sampler = sparco.sampler.Sampler(**config['sampler'])
for c in config['nets']:
  c['sampler'] = sparco.sampler.Sampler(**pfacets.merge(config['sampler'], c['sampler']))

# TODO need to manage this better
if config['run_ladder']:
  run_coder(config)
else:
  for net in config['nets']:
    conf = copy.copy(config)
    conf['nets'] = net
    if net.has_key('trace'):
      conf['trace'] = pfacets.merge(conf['trace'], net.pop('trace'))
    run_coder(conf)
