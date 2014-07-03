#!/usr/bin/env python
from IPython import embed

import argparse
import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.realpath(__file__), '..', '..')))
# TODO fix this-- the issue is that tokyo is not on the load path
sys.path.append(
    os.path.normpath(os.path.join(os.path.realpath(__file__), '..', '..', 'sparco', 'qn')))

import sparco.sptools as sptools

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--channels',
    help='comma-separated string of channel ranges and individual values')
arg_parser.add_argument('-C', '--local-config-path',
    help='path to a python module containing a configuration dictionary')
arg_parser.add_argument('-i', '--input-path',
    help='path to directory containing input files')
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


# build configuration

args = parse_args()
cli_config = sptools.expand_args(args, {
    'input_path': ['db', 'input_path'],
    'output_path': ['trace', 'output_path'],
    'snapshot_interval': ['trace', 'snapshot_interval']
    })
local_config = sptools.load_config(path = args.local_config_path,
    default_name='.sparcorc').config

config = sptools.merge(local_config, cli_config)

if args.produce_output or args.output_path:
  import sparco.trace
  sparco.trace.configure(config['trace'])

db = sparco.db.DB(**config['db'])
for c in config['nets']:
  c['db'] = db
sc = sparco.sparse_coder.SparseCoder(config['nets'])
sc.run()
