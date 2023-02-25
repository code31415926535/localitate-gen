import json
import random
import torch
from typing import List
import torch.nn.functional as F
import utils

class MLP:
  block_size = 4
  batch_size = 64
  embedding_size = 10
  layer_one_size = 350
  learning_rate = 0.12
  learning_rate_decay_after = 30000
  training_iterations = 65000

  def __init__(self, load=False):
    if load:
      self.load_model()

  # TODO: clean this up a bit
  def train(self, data, save=False):
    self._generate_vocab(data)
    self._setup_hidden_layer()
    self.iterations = []
    self.devLosses = []
    self.trainLosses = []
    (Xtrain, Ytrain), (Xdev, Ydev) = self._generate_training_data(data)
    print('Training data size:', Xtrain.shape[0])
    print('Vocab size:', len(self.stoi))
    print('Block size:', self.block_size)
    print('Batch size:', self.batch_size)
    print('Embedding size:', self.embedding_size)
    print('Layer one size:', self.layer_one_size)
    print('Learning rate:', self.learning_rate)
    print('Learning rate decay after:', self.learning_rate_decay_after)
    print('Training iterations:', self.training_iterations)
    for p in self.parameters():
      p.requires_grad = True
    lr = self.learning_rate
    for i in range(self.training_iterations):
      # Minibatch
      ix = torch.randint(0, Xtrain.shape[0], (self.batch_size,))
      loss = self._forward_pass(Xtrain[ix], Ytrain[ix])
      self._backward_pass(loss, lr)
      if (i % 1000) == 0:
        with torch.no_grad():
          devLoss = self._forward_pass(Xdev, Ydev)
          trainLoss = self._forward_pass(Xtrain, Ytrain)
          self.iterations.append(i)
          self.devLosses.append(devLoss.item())
          self.trainLosses.append(trainLoss.item())
          print(f'Dev loss {devLoss.item():.3f}, train loss {trainLoss.item():.3f}')
      if (i % self.learning_rate_decay_after) == 0 and (i > 0):
        lr /= 10
        print('Learning rate decayed to', lr)

    if save:
      self.save_model()

  def stats(self):
    return {
      'block_size': self.block_size,
      'batch_size': self.batch_size,
      'embedding_size': self.embedding_size,
      'layer_one_size': self.layer_one_size,
      'learning_rate': self.learning_rate,
      'learning_rate_decay_after': self.learning_rate_decay_after,
      'training_iterations': self.training_iterations,
      'training': {
        'iterations': self.iterations, 
        'devLosses': self.devLosses, 
        'trainLosses': self.trainLosses,
      },
      'embedding': self.C.tolist(),
    }

  def sample(self, prefix: str = ''):
    out = []
    context = [0] * self.block_size
    for ch in prefix:
      ix = self.stoi[ch]
      out.append(ch)
      context = context[1:] + [ix]
    for _ in range(100):
      ix = self._forward_pass(torch.tensor([context]), train=False)
      ch = self.itos[ix.item()]
      if ch == '#':
        break
      out.append(ch)
      context = context[1:] + [ix]
    return ''.join(out)

  def _forward_pass(self, dataset, targets = [], train = True):
    emb = self.C[dataset]
    flat_emb = emb.view(-1, self.block_size * self.embedding_size)
    h = torch.tanh(flat_emb @ self.W1 + self.B1)
    logits = h @ self.W2 + self.B2
    if train:
      loss = F.cross_entropy(logits, targets)
      return loss
    else:
      probs = F.softmax(logits, dim=1)
      return torch.multinomial(probs, 1)
  
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
    g = torch.Generator().manual_seed(42)
    self.C = torch.randn(len(self.stoi), self.embedding_size, generator=g)
    self.W1 = torch.randn((self.block_size * self.embedding_size, self.layer_one_size), generator=g) * (5/3)/(self.block_size * self.embedding_size)**0.5
    self.B1 = torch.randn(self.layer_one_size, generator=g) * 0.01
    self.W2 = torch.randn((self.layer_one_size, len(self.stoi)), generator=g) * 0.01
    self.B2 = torch.randn(len(self.stoi), generator=g) * 0
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
    with open('stats.json', 'w') as f:
      json.dump(self.stats(), f, indent=2)

  def parameters(self):
    return [self.C, self.W1, self.B1, self.W2, self.B2]