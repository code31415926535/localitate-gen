from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox

def plot_learning_rate(stats: dict, ax):
    training = stats['training']
    ax.set_title('Loss over time')
    ax.set_ylabel('loss')
    ax.set_xlabel('iteration')
    ax.plot(training['iterations'], training['devLosses'], label='dev')
    ax.plot(training['iterations'], training['trainLosses'], label='train')
    ax.legend()

def plot_embedding(stats: dict, ax):
    embedding = stats['embedding']
    flat_embedding = [item for sublist in embedding for item in sublist]
    mn, mx = min(flat_embedding), max(flat_embedding)
    ax.set_title('Embedding')
    ax.set_ylabel('letter')
    ax.set_xlabel('embedding dimension')
    ax.imshow(embedding, cmap='hot', vmin=mn, vmax=mx, interpolation='nearest')

def plot_scatter_embedding(stats: dict, ax):
    embedding = stats['embedding']
    ax.set_title('Embedding')
    firstDim = 1
    secondDim = 2
    ax.set_ylabel(f'embedding dimension {secondDim}')
    ax.set_xlabel(f'embedding dimension {firstDim}')
    dimOne = [x[firstDim] for x in embedding]
    dimTwo = [x[secondDim] for x in embedding]
    vocab = stats['vocab']
    ax.scatter(dimOne, dimTwo)
    for i, txt in enumerate(vocab):
      ax.annotate(txt, (dimOne[i] + 0.1, dimTwo[i]))

def plot_stats(stats):
    gs_kw = dict(width_ratios=[1.5, 1], height_ratios=[1, 2])
    fig, axd = plt.subplot_mosaic([['lr', 'emb'], ['scatter', 'emb']], gridspec_kw=gs_kw, layout="constrained")
    plot_learning_rate(stats, axd['lr'])
    plot_embedding(stats, axd['emb'])
    plot_scatter_embedding(stats, axd['scatter'])
    plt.show()