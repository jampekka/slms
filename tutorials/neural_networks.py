import marimo

__generated_with = "0.6.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import utils as U
    import altair as alt
    import pandas as pd
    import polars as pl
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    return F, U, alt, mo, nn, np, pd, pl, torch


@app.cell
def __(mo):
    mo.md(
        rf"""
        # Neural network models

        In this notebook, we'll build a very simple **neural network** language model. Conceptually the resulting model is very close to current popular LLMs but the neural network in them is vastly larger (have often billions of parameters vs our few parameters) and have some more constraints in their structure (e.g. in models based on the Transformer architecture).

        Let's go back to the Happy Birthday lyrics from the [basics](?file=basics), using the word tokenizer to keep things intuitive:
        """
    )
    return


@app.cell
def __(mo):
    _happy_birthday_text = """
     Happy birthday to you 
     Happy birthday to you 
     Happy birthday dear Dave 
     Happy birthday to you 
    """

    _blowin_text = """
     Yes, and how many roads must a man walk down, before you call him a man? 
     And how many seas must a white dove sail, before she sleeps in the sand? 
     Yes, and how many times must the cannonballs fly, before they're forever banned? 

     Yes, and how many years must a mountain exist, before it is washed to the sea? 
     And how many years can some people exist, before they're allowed to be free? 
     Yes, and how many times can a man turn his head, and pretend that he just doesn't see? 

     Yes, and how many times must a man look up, before he can see the sky? 
     And how many ears must one man have, before he can hear people cry? 
     Yes, and how many deaths will it take 'til he knows, that too many people have died? 
    """

    corpus_selections = {
        "Happy Birthday": _happy_birthday_text,
        "Blowin' in the wind": _blowin_text
    }

    corpus_selector = mo.ui.dropdown(corpus_selections, value="Happy Birthday")
    return corpus_selections, corpus_selector


@app.cell
def __(U, corpus_selector):
    tokenizer = U.HackyWordTokenizer()

    corpus_name = corpus_selector.selected_key
    corpus_text = corpus_selector.value

    corpus_words = tokenizer(corpus_text)

    vocabulary = U.corpus_to_vocabulary(corpus_words)
    vocabulary_size = len(vocabulary)
    word_vocab_pos = {w: i for i, w in enumerate(vocabulary)}

    U.tokens_out(corpus_words, tokenizer)
    return (
        corpus_name,
        corpus_text,
        corpus_words,
        tokenizer,
        vocabulary,
        vocabulary_size,
        word_vocab_pos,
    )


@app.cell
def __(mo):
    mo.md(rf"And let's do the simple context length one model, where we predict the next word just based on the current word:")
    return


@app.cell
def __(U, corpus_words):
    next_words = {}
    for i in range(len(corpus_words)-1):
        word = corpus_words[i]
        next_word = corpus_words[i+1]
        if word not in next_words:
            next_words[word] = []
        next_words[word].append(next_word)
    context_length = 1
    next_words_count = U.get_next_token_table(corpus_words, context_length=context_length)
    U.python_out(next_words)
    return context_length, i, next_word, next_words, next_words_count, word


@app.cell
def __(mo):
    mo.md(rf"We can also visualize this model as a, well, network. On the left we have the current word, on the right we have the next word, and a line between them means that the next word can follow from the current word, and the strength of the line how likely it is to follow:")
    return


@app.cell
def __(alt, mo, next_words_count, torch, vocabulary, vocabulary_size):
    import itertools

    _current_to_next = list(itertools.permutations(range(len(vocabulary)), 2))

    _follower_counts = torch.zeros((len(vocabulary), len(vocabulary)))
    for _i in range(vocabulary_size):
        _current = (vocabulary[_i], )
        for _j in range(vocabulary_size):
            _next = vocabulary[_j]
            _follower_counts[_i,_j] = next_words_count[_current].get(_next, 0)

    connections = _connections = _follower_counts/_follower_counts.sum(dim=0)

    def plot_nn_layer(connections, left_labels=[], right_labels=[]):
        #froms, tos = np.nonzero(connections)
        #w = connections[x, y]

        leftpos = 0
        rightpos = 1

        chart = alt.layer()
        for x in range(connections.shape[0]):
            for y in range(connections.shape[1]):
                w = float(connections[x, y])
                line = alt.Chart().mark_line(size=3).encode(
                    x=alt.datum(leftpos, axis=None),
                    y=alt.datum(-x, axis=None),
                    x2=alt.datum(rightpos),
                    y2=alt.datum(-y),
                    opacity=alt.value(w*0.8)
                )

                chart += line

        for x, text in left_labels:
            chart += alt.Chart().mark_text(align='right', fontSize=16).encode(
                x=alt.datum(leftpos),
                y=alt.datum(-x),
                text=alt.value(text),
            )

        for x, text in right_labels:
            chart += alt.Chart().mark_text(align='left', fontSize=16).encode(
                x=alt.datum(rightpos),
                y=alt.datum(-x),
                text=alt.value(text),
            )

        chart = chart.configure_axis(grid=False).configure_view(strokeWidth=0)
        return mo.ui.altair_chart(chart)

    _labels = list(enumerate(map(repr, vocabulary)))
    plot_nn_layer(_connections, _labels, _labels)

    #_x, _y = zip(*_current_to_next)
    #
    #_data = pd.DataFrame({'x': _x, 'y': _y})
    #print(_data)
    #_chart = alt.Chart(_data).mark_point().encode(x='x', y='y')
    #mo.ui.altair_chart(_chart)
    return connections, itertools, plot_nn_layer


@app.cell
def __(mo):
    mo.md(rf"The same network can be also represented as weights, where the rows mean the current word, and columns the next word:")
    return


@app.cell
def __(connections, mo, pd, vocabulary):
    _voc = list(map(repr, vocabulary))
    _df = pd.DataFrame(connections, columns=_voc, index=_voc)
    mo.ui.table(_df, selection=None)
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        And here we have it! A neural network language model! It's admittedly very simple one, and what would be called a single layer neural network.

        We "trained" the neural network by explicitly calculating the followers. But typically neural networks are trained using optimization, i.e. we see some examples of previous and next words, and we try to make the neural network to match those examples as well as possible.

        We can train our model this way too. Typically the weights are initialized to some random values, and then the optimization algorithm starts to "tweak" them to get the predictions match the data better.
        """
    )
    return


@app.cell
def __(
    F,
    U,
    context_length,
    corpus_words,
    mo,
    nn,
    torch,
    vocabulary_size,
    word_vocab_pos,
):
    torch.manual_seed(1337)
    from copy import deepcopy

    class SimpleNnLm(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            # This would be faster with nn.Embedding, so we wouldn't have
            # to one_hot. But let's do this and one-hot for clarity
            self.mlp = torch.nn.Linear(vocab_size, vocab_size, bias=False)

        def forward(self, input):
            input = F.one_hot(input, self.vocab_size).float()
            logits = self.mlp(input)

            return logits

        def generate(self, context):
            context = torch.tensor(context)
            while True:
                yield context
                logits = self(context)
                context = torch.argmax(logits)

    corpus_ids = [word_vocab_pos[w] for w in corpus_words]

    dataset = torch.as_tensor(list(U.get_ngrams(corpus_ids, context_length+1)))
    contexts = dataset[:,:-1]
    targets = dataset[:,-1]

    _model = SimpleNnLm(vocabulary_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(_model.parameters(), lr=1e-1)

    losses = []
    model_steps = []
    n_iterations = 100
    for _i in range(n_iterations):
        predictions = _model(contexts).squeeze(1)
        loss = criterion(predictions, targets.view(-1))
        model_steps.append(deepcopy(_model))
        losses.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    iter_selector = mo.ui.number(0, n_iterations, value=5)
    return (
        SimpleNnLm,
        contexts,
        corpus_ids,
        criterion,
        dataset,
        deepcopy,
        iter_selector,
        loss,
        losses,
        model_steps,
        n_iterations,
        optimizer,
        predictions,
        targets,
    )


@app.cell
def __(corpus_name, mo):
    mo.md(rf"Below we can see the results of learning from the lyrics of {corpus_name}.")
    return


@app.cell
def __(
    U,
    alt,
    corpus_ids,
    iter_selector,
    itertools,
    losses,
    mo,
    model_steps,
    n_iterations,
    pd,
    tokenizer,
    vocabulary,
):


    model_step = iter_selector.value

    model = model_steps[model_step]

    _chart = alt.Chart(pd.DataFrame({'Fit iteration': range(n_iterations), 'Loss': losses})
                      ).mark_line().encode(
                        x="Fit iteration",
                        y="Loss"
                      )
    loss_chart = mo.ui.altair_chart(_chart)

    _max_len = len(corpus_ids)
    _gen_ids = itertools.islice(model.generate(corpus_ids[0]), _max_len)
    generated = [vocabulary[id] for id in _gen_ids]
    generated = U.tokens_out(generated, tokenizer)

    generated

    #mo.vstack((
    #    iter_selector,
    #    generated,
    #))

    return generated, loss_chart, model, model_step


if __name__ == "__main__":
    app.run()
