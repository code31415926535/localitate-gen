import argparse
from mlp import MLP
from matplotlib import pyplot as plt
import utils

def train(plot=False):
  # data = utils.read_magyarorszag()
  data = utils.read_csv()
  model = MLP()
  model.train(data, save=True)
  stats = model.stats()
  training = stats['training']
  if plot:
    plt.plot(training['iterations'], training['devLosses'], label='dev')
    plt.plot(training['iterations'], training['trainLosses'], label='train')
    plt.show()

def sample(prefix: str, count: int):
  model = MLP(load=True)
  for _ in range(count):
    result = model.sample(prefix)
    print(result)

if __name__ == '__main__':
  argparse = argparse.ArgumentParser()
  argparse.add_argument('--train', action='store_true')
  argparse.add_argument('--plot', action='store_true')
  argparse.add_argument('--sample', action='store_true')
  argparse.add_argument('--prefix', type=str, default='')
  argparse.add_argument('--count', type=int, default=20)
  args = argparse.parse_args()
  if args.train:
    train(args.plot)
  if args.sample:
    sample(args.prefix, args.count)
