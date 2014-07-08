#!/usr/bin/env python
from IPython import embed

import argparse
import os
import sys

import pfacets

########### SET UP PATH / PARSE COMMAND-LINE ARGUMENTS
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__),  '..')))
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'sparco', 'qn')))

import sparco

# TODO fix this-- the issue is that tokyo is not on the load path

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
arg_parser.add_argument('-O', '--produce-output', action='store_true',
    help='enables output of logging and basis snapshots; not needed if output_path provided')
arg_parser.add_argument('-o', '--output-path',
    help='path to directory containing output files')
arg_parser.add_argument('-s', '--snapshot-interval', default=100, type=int,
    help='number of iterations between basis snapshots')

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
    'output_path': ['trace', 'output_path'],
    'snapshot_interval': ['trace', 'snapshot_interval']
    })
config_module = pfacets.load_local_module(path = args.local_config_path,
    default_name='.sparcorc')
local_config = config_module.config if config_module else {}
config = pfacets.merge(local_config, cli_config)

########### SPARSE_CODER_OBJECT

sampler = sparco.sampler.Sampler(**config['sampler'])
[c.setdefault('sampler', sampler) for c in config['nets']]
sc = sparco.sparse_coder.SparseCoder(config['nets'])

########### CONFIGURE OUTPUT

if args.produce_output or args.output_path:
  import sparco.trace
  sparco.trace.configure(sc, config['trace'])

########### RUN SparseCoder

sc.run()
