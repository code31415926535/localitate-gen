from typing import List
import torch
import torch.nn.functional as F
import utils

class Model:
  block_size = 3
  embedding_size = 2
  layer_one_size = 100

  def __init__(self, stoi: dict, itos: dict):
    self.stoi = stoi
    self.itos = itos
    print('Vocab size:', len(stoi))
    print('Block size:', self.block_size)
    print('Embedding size:', self.embedding_size)
    print('Layer one size:', self.layer_one_size)
    self._setup_hidden_layer()

  def train(self, data):
    X, Y = self._generate_training_data(data)
    emb = self.C[X]
    flat_emb = emb.view(-1, self.block_size * self.embedding_size)
    h = torch.tanh(flat_emb @ self.W1 + self.B1)
    logits = h @ self.W2 + self.B2
    loss = F.cross_entropy(logits, Y)
    print(loss)

  def _generate_training_data(self, data: List[str]):
    X = []
    Y = []
    for w in data:
      context = [0] * self.block_size
      for ch in w + '#':
        ix = self.stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join([self.itos[i] for i in context]), '->', self.itos[ix])
        context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

  def _setup_hidden_layer(self):
    g = torch.Generator().manual_seed(42)
    self.C = torch.randn(len(self.stoi), self.embedding_size, generator=g)
    self.W1 = torch.randn((self.block_size * self.embedding_size, self.layer_one_size), generator=g)
    self.B1 = torch.randn(self.layer_one_size, generator=g)  
    self.W2 = torch.randn((self.layer_one_size, len(self.stoi)), generator=g)
    self.B2 = torch.randn(len(self.stoi), generator=g)
    print('Paramters:', sum(p.numel() for p in self.parameters()))

  def parameters(self):
    return [self.C, self.W1, self.B1, self.W2, self.B2]

if __name__ == '__main__':
  data = utils.read_csv()
  chars = utils.get_chars(data)
  stoi = {char: i+1 for i, char in enumerate(chars)}
  stoi['#'] = 0
  itos = {i: char for char, i in stoi.items()}
  model = Model(stoi, itos)
  model.train(data[:10])
