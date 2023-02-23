import random
import argparse
from typing import List
import torch
import torch.nn.functional as F
import utils

class Model:
  block_size = 4
  batch_size = 64
  embedding_size = 10
  layer_one_size = 350
  learning_rate = 0.12
  learning_rate_decay_after = 30000
  training_iterations = 75000

  def __init__(self, load=False):
    print('Block size:', self.block_size)
    print('Embedding size:', self.embedding_size)
    print('Layer one size:', self.layer_one_size)
    print('Learning rate:', self.learning_rate)
    print('Learning rate decay:', self.learning_rate_decay_after)
    print('Training iterations:', self.training_iterations)
    if load:
      self.load_model()

  def train(self, data, save=False):
    self._generate_vocab(data)
    self._setup_hidden_layer()
    (Xtrain, Ytrain), (Xdev, Ydev) = self._generate_training_data(data)
    print('Training data size:', Xtrain.shape[0])
    print('Vocab size:', len(self.stoi))
    for p in self.parameters():
      p.requires_grad = True
    lr = self.learning_rate
    for i in range(self.training_iterations):
      # Minibatch
      ix = torch.randint(0, Xtrain.shape[0], (self.batch_size,))
      loss = self._forward_pass(Xtrain[ix], Ytrain[ix])
      self._backward_pass(loss, lr)
      if (i % 5000) == 0:
        with torch.no_grad():
          devLoss = self._forward_pass(Xdev, Ydev)
          trainLoss = self._forward_pass(Xtrain, Ytrain)
          print(f'Dev loss {devLoss.item():.2f}, train loss {trainLoss.item():.2f}')
      if (i % self.learning_rate_decay_after) == 0 and (i > 0):
        lr /= 10
        print('Learning rate decayed to', lr)

    if save:
      self.save_model()

  def sample(self, prefix: str = ''):
    out = []
    context = [0] * self.block_size
    for ch in prefix:
      ix = self.stoi[ch]
      out.append(ch)
      context = context[1:] + [ix]
    for _ in range(100):
      emb = self.C[torch.tensor([context])]
      flat_emb = emb.view(-1, self.block_size * self.embedding_size)
      h = torch.tanh(flat_emb @ self.W1 + self.B1)
      logits = h @ self.W2 + self.B2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, 1)
      ch = self.itos[ix.item()]
      if ch == '#':
        break
      out.append(ch)
      context = context[1:] + [ix]
    return ''.join(out)

  def _forward_pass(self, dataset: List[str] = [], targets: List[str] = []):
    emb = self.C[dataset]
    flat_emb = emb.view(-1, self.block_size * self.embedding_size)
    h = torch.tanh(flat_emb @ self.W1 + self.B1)
    logits = h @ self.W2 + self.B2
    loss = F.cross_entropy(logits, targets)
    return loss
  
  def _backward_pass(self, loss, lr):
    for p in self.parameters():
      p.grad = None
    loss.backward()
    for p in self.parameters():
      p.data -= lr * p.grad

  def _generate_training_data(self, data: List[str]):
    random.shuffle(data)
    splitTrain = int(len(data) * 0.9)
    train = self._generate_split(data[:splitTrain])
    dev = self._generate_split(data[splitTrain:])
    return train, dev

  def _generate_split(self, data: List[str]):
    X = []
    Y = []
    for w in data:
      context = [0] * self.block_size
      for ch in w + '#':
        ix = self.stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return (X, Y)

  def _generate_vocab(self, data):
    chars = utils.get_chars(data)
    self.stoi = {char: i+1 for i, char in enumerate(chars)}
    self.stoi['#'] = 0
    self.itos = {i: char for char, i in self.stoi.items()}

  def _setup_hidden_layer(self):
    g = torch.Generator()
    self.C = torch.randn(len(self.stoi), self.embedding_size, generator=g)
    self.W1 = torch.randn((self.block_size * self.embedding_size, self.layer_one_size), generator=g)
    self.B1 = torch.randn(self.layer_one_size, generator=g)  
    self.W2 = torch.randn((self.layer_one_size, len(self.stoi)), generator=g)
    self.B2 = torch.randn(len(self.stoi), generator=g)
    print('Paramters:', sum(p.numel() for p in self.parameters()))

  def load_model(self):
    self.C, self.W1, self.B1, self.W2, self.B2 = torch.load('model.pt')
    with open('vocab.txt') as f:
      self.stoi = {}
      self.itos = {}
      for i, line in enumerate(f):
        char = line.split('\n')[0]
        self.stoi[char] = i
        self.itos[i] = char

  def save_model(self):
    torch.save(self.parameters(), 'model.pt')
    with open('vocab.txt', 'w') as f:
      for i in range(len(self.stoi)):
        f.write(f'{self.itos[i]}\n')

  def parameters(self):
    return [self.C, self.W1, self.B1, self.W2, self.B2]

def train():
  data = utils.read_csv()
  model = Model()
  model.train(data, save=True)

def sample(prefix: str, count: int):
  model = Model(load=True)
  for _ in range(count):
    result = model.sample(prefix)
    print(result)

if __name__ == '__main__':
  argparse = argparse.ArgumentParser()
  argparse.add_argument('--train', action='store_true')
  argparse.add_argument('--sample', action='store_true')
  argparse.add_argument('--prefix', type=str, default='')
  argparse.add_argument('--count', type=int, default=20)
  args = argparse.parse_args()
  if args.train:
    train()
  if args.sample:
    sample(args.prefix, args.count)
