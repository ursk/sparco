import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', '--channels',
    help='comma-separated string of channel ranges and individual values')
arg_parser.add_argument('-i', '--input-directory',
    help='path to directory containing input files')
arg_parser.add_argument('-o', '--output-directory',
    help='path to directory containing output files')

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
