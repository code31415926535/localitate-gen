import argparse
from mlp import MLP
from plots import plot_stats
import utils

def train(plot=False):
  # data = utils.read_magyarorszag()
  data = utils.read_csv()
  model = MLP()
  model.train(data, save=True)
  stats = model.stats()
  if plot:
    plot_stats(stats)

def sample(prefix: str, count: int):
  model = MLP(load=True)
  for _ in range(count):
    result = model.sample(prefix)
    print(result)

def stats():
  stats = utils.read_stats()
  plot_stats(stats)

if __name__ == '__main__':
  argparse = argparse.ArgumentParser()
  subparsers = argparse.add_subparsers(dest='subparser_name')

  train_parser = subparsers.add_parser('train', help='train the model')
  train_parser.add_argument('--plot', action='store_true')

  sample_parser = subparsers.add_parser('sample', help='sample from the model')
  sample_parser.add_argument('--prefix', type=str, default='')
  sample_parser.add_argument('--count', type=int, default=20)

  stats_parser = subparsers.add_parser('stats', help='show training stats')

  args = argparse.parse_args()
  if args.subparser_name == 'train':
    train(args.plot)
  if args.subparser_name == 'sample':
    sample(args.prefix, args.count)
  if args.subparser_name == 'stats':
    stats()
