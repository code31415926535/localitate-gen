# Localitate Gen

Tool to generate Romanian sounding place names.

## Sampling the model

```bash
python3.7 main.py sample --count 10
vladesti
furdului
albiu
ciobana
dobra
dornenu
ludosteni
carjmeschelmoceni
popesti
valea lun
```

You can also use completion, by giving a suffix to the sample command

```bash
main.py sample --prefix ma --count 5
macesti
magura roman
mandrusului
mazesti
malea mlan
```

## Training

The model can be training using the `train` command. An optional `plot` option can be added to show the plot.

```bash
python3.7 main.py train --plot
Paramters: 26624
Training data size: 116433
Vocab size: 34
Block size: 4
Batch size: 64
Embedding size: 10
Layer one size: 350
Learning rate: 0.12
Learning rate decay after: 30000
Training iterations: 65000
Dev loss 3.393, train loss 3.391
Dev loss 2.069, train loss 2.071
...
Dev loss 1.690, train loss 1.486
```

![Learning stats](/docs/stats.png)
