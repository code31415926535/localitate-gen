import random
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
    ax.set_title('Embedding')
    ax.set_ylabel('letter')
    ax.set_xlabel('embedding dimension')
    vocabInOrder = [x[0] for x in sorted(stats['vocab'].items(), key=lambda item: item[1])]
    ax.set_yticks(range(len(embedding)), labels=vocabInOrder)
    ax.set_xticks(range(len(embedding[0])))
    ax.imshow(embedding, interpolation='nearest')

def plot_selected(stats: dict, ax, selected = [], title = ''):
    embedding = stats['embedding']
    ax.set_title(title)
    ax.set_ylabel('embedding dimension')
    ax.set_xlabel('letter')
    letterIndices = [stats['vocab'][letter] for letter in selected]
    dimension = []
    value = []
    color = []
    for cnt, i in enumerate(letterIndices):
        c = cnt / len(letterIndices)
        for j in range(len(embedding[i])):
            dimension.append(j)
            value.append(embedding[i][j])
            color.append(c)
    scatter = ax.scatter(value, dimension, c=color)
    ax.legend(handles=scatter.legend_elements()[0], labels=selected)

def plot_stats(stats):
    gs_kw = dict(width_ratios=[1.5, 1], height_ratios=[1, 2])
    fig, axd = plt.subplot_mosaic([['lr', 'emb'], ['scatter', 'emb']], gridspec_kw=gs_kw, layout="constrained")
    plot_learning_rate(stats, axd['lr'])
    plot_embedding(stats, axd['emb'])
    # plot_selected(stats, axd['scatter'], title='Vowels', selected=['a', 'e', 'i', 'o', 'u'])
    # plot_selected(stats, axd['scatter'], title='Consonants', selected=['b', 'c', 'd', 'f', 'g', 'h'])
    plot_selected(stats, axd['scatter'], title='Mixed', selected=['a', 'e', 'i', 'o', 'b', 'c', 'd', 'f'])
    plt.show()