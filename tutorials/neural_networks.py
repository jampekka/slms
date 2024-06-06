import marimo

__generated_with = "0.6.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import utils as U
    import altair as alt
    import matplotlib.pyplot as plt
    import pandas as pd
    import polars as pl
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    return F, U, alt, mo, nn, np, pd, pl, plt, torch


@app.cell
def __():
    # Define options

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

    _blowin_text = """
     Yes, and how many roads must a man walk down, before you call him a man? 
     And how many seas must a white dove sail, before she sleeps in the sand? 
     Yes, and how many times must the cannonballs fly, before they're forever banned? 
    """

    corpus_selections = {
        "Happy Birthday": _happy_birthday_text,
        "Blowin' in the wind": _blowin_text
    }

    n_iterations = 100
    context_length = 1

    return context_length, corpus_selections, n_iterations


@app.cell
def __(corpus_selections, mo, n_iterations):
    # Define controls
    corpus_selector = mo.ui.dropdown(corpus_selections, value="Happy Birthday")
    iter_selector = mo.ui.slider(0, n_iterations-1, value=20,
                                 show_value=True,
                                 label="Training iteration",
                                 full_width=True,
                                 debounce=True,
                                )
    generation_seed_selector = mo.ui.slider(start=1, value=1, stop=10,
                                            full_width=False, show_value=True,
                                            label="Variation (random seed)"
                                           )
    return corpus_selector, generation_seed_selector, iter_selector


@app.cell
def __(corpus_selector):
    # Define variables derived from controls
    corpus_name = corpus_selector.selected_key
    corpus_text = corpus_selector.value
    return corpus_name, corpus_text


@app.cell
def __(generation_seed_selector):
    generation_seed = generation_seed_selector.value
    return generation_seed,


@app.cell
def __(U, corpus_text):
    # Define variables derived from control variables
    tokenizer = U.HackyWordTokenizer()

    corpus_words = tokenizer(corpus_text)

    vocabulary = U.corpus_to_vocabulary(corpus_words)
    vocabulary_size = len(vocabulary)
    word_vocab_pos = {w: i for i, w in enumerate(vocabulary)}
    return (
        corpus_words,
        tokenizer,
        vocabulary,
        vocabulary_size,
        word_vocab_pos,
    )


@app.cell
def __(corpus_name, mo):
    mo.md(
        rf"""
        # Neural network models

        In this notebook, we'll build a very simple **neural network** language model. Conceptually the resulting model is very close to current popular LLMs but the neural network in them is vastly larger (have often billions of parameters vs our few parameters) and have some more constraints in their structure (e.g. in models based on the Transformer architecture).

        Let's analyze lyrics of {corpus_name}. We use the simple word tokenizer to keep things more intuitive. The neural network works exactly the same with other tokenizers.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __(U, corpus_words, tokenizer):
    U.tokens_out(corpus_words, tokenizer)
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## The basic model as a neural network

        And let's do the simple context length one model, where we predict the next word just based on the current word:
        """
    )
    return


@app.cell
def __(U, context_length, corpus_words):
    next_words = {}
    for i in range(len(corpus_words)-1):
        word = corpus_words[i]
        next_word = corpus_words[i+1]
        if word not in next_words:
            next_words[word] = []
        next_words[word].append(next_word)

    next_words_count = U.get_next_token_table(corpus_words, context_length=context_length)
    U.python_out(next_words)
    return i, next_word, next_words, next_words_count, word


@app.cell
def __(mo):
    mo.md(
        rf"""
        We can also visualize this model as a, well, network. In the graph, on the left we have the current word, on the right we have the next word, and a line between them means that the next word can follow from the current word, and the strength of the line how likely it is to follow.

        In the table we have the same information, but as numbers.
        """
    )
    return


@app.cell
def __(
    mo,
    next_words_count,
    np,
    pd,
    plt,
    torch,
    vocabulary,
    vocabulary_size,
):
    import itertools

    _current_to_next = list(itertools.permutations(range(len(vocabulary)), 2))

    _follower_counts = torch.zeros((len(vocabulary), len(vocabulary)))
    for _i in range(vocabulary_size):
        _current = (vocabulary[_i], )
        for _j in range(vocabulary_size):
            _next = vocabulary[_j]
            _follower_counts[_i,_j] = next_words_count[_current].get(_next, 0)

    connections = _connections = _follower_counts/_follower_counts.sum(dim=0).reshape(-1, 1)

    def plot_nn_layer(connections, left_labels, right_labels):
        from matplotlib.collections import LineCollection
        #froms, tos = np.nonzero(connections)
        #w = connections[x, y]

        leftpos = 0
        rightpos = 1

        nx, ny = connections.shape
        #xs, ys = np.indices((nx, ny), sparse=True)
        xs, ys = map(np.ravel, np.mgrid[0:nx, 0:ny])
        weights = connections[xs, ys]
        
        token_positions = np.vstack((xs, ys)).T
        n = len(token_positions)
        side_positions = np.vstack((np.repeat(leftpos, n), np.repeat(rightpos, n))).T
        segs = np.dstack((side_positions, token_positions))
        lc = LineCollection(segs, color='black', alpha=weights*0.8)
        ax = plt.gca()
        ax.add_collection(lc)

        tax = ax.twinx()

        for a in [ax, tax]:
            a.set_yticks(ticks=np.arange(nx), labels=left_labels)
            a.set_ylim(max(*connections.shape), -1)
            for spine in a.spines.values():
                spine.set_visible(False)

        ax.set_xticks([])
        
        return ax

    def nn_weights_table(connections, left_labels, right_labels):
        df = pd.DataFrame(connections,
                          columns=left_labels,
                          index=left_labels
        )
        return mo.ui.table(df, selection=None)

    _labels = list(map(repr, vocabulary))

    mo.ui.tabs({
        "Weight graph": plot_nn_layer(_connections, _labels, _labels),
        "Weight table": nn_weights_table(_connections, _labels, _labels)
        }
    )
    return connections, itertools, nn_weights_table, plot_nn_layer


@app.cell
def __(mo):
    mo.md(
        rf"""
        And here we have it! A neural network language model! It's admittedly very simple one, and what would be called a single layer neural network.

        ## Neural network training

        We "trained" the neural network by explicitly calculating the followers. But typically neural networks are trained using optimization, i.e. we see some examples of previous and next words, and we try to make the neural network to match those examples as well as possible.

        We can train our model this way too. Typically the weights are initialized to some random values, and then the optimization algorithm starts to "tweak" them to get the predictions match the data better.
        """
    )
    return


@app.cell
def __(F, nn, torch):
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

        def next_probabilities(self, input):
            logits = self(input)
            return F.softmax(logits, dim=-1)

        def generate(self, context, seed=None):
            generator = None
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(seed)
            context = torch.tensor(context)
            while True:
                yield context
                logits = self(context)
                probs = F.softmax(logits, dim=0)
                context = torch.multinomial(probs, 1)[0]
    return SimpleNnLm,


@app.cell
def __(
    SimpleNnLm,
    U,
    context_length,
    corpus_words,
    n_iterations,
    nn,
    torch,
    vocabulary_size,
    word_vocab_pos,
):
    torch.manual_seed(1337)
    from copy import deepcopy

    corpus_ids = [word_vocab_pos[w] for w in corpus_words]

    dataset = torch.as_tensor(list(U.get_ngrams(corpus_ids, context_length+1)))
    contexts = dataset[:,:-1]
    targets = dataset[:,-1]

    _model = SimpleNnLm(vocabulary_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(_model.parameters(), lr=5e-2)

    losses = []
    model_steps = []
    for _i in range(n_iterations + 1):
        predictions = _model(contexts).squeeze(1)
        loss = criterion(predictions, targets.view(-1))
        model_steps.append(deepcopy(_model))
        losses.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (
        contexts,
        corpus_ids,
        criterion,
        dataset,
        deepcopy,
        loss,
        losses,
        model_steps,
        optimizer,
        predictions,
        targets,
    )


@app.cell
def __(corpus_name, corpus_selector, iter_selector, mo):
    mo.md(
        rf"""
        Below we can see the results of learning from the lyrics of {corpus_name}. You can change how many iterations the model was trained with. With zero iterations the model should make no sense, but with more iterations it starts to get a bit better, especially with increased context length! With enough iterations, it should match the simple frequency table model we started with.

        The Happy Birthday lyrics may not be that interesting, so try others: {corpus_selector}!

        Note that the whole document changes to the lyrics you select, so scroll back to the beginning too!

        {iter_selector}
        """
    )
    return


@app.cell
def __(
    iter_selector,
    mo,
    model_steps,
    nn_weights_table,
    plot_nn_layer,
    torch,
    vocabulary,
    vocabulary_size,
):


    model_step = iter_selector.value
    model = model_steps[model_step]

    next_probs = model.next_probabilities(torch.arange(vocabulary_size)).detach()
    _labels = list(map(repr, vocabulary))

    weights_ui = mo.ui.tabs({
        "Weight graph": plot_nn_layer(next_probs, _labels, _labels),
        "Weight table": nn_weights_table(next_probs, _labels, _labels)
        }
    )
    return model, model_step, next_probs, weights_ui


@app.cell
def __(losses, mo, model_step, n_iterations, plt):
    _, _ax = plt.subplots()
    _ax.plot(range(n_iterations+1), losses, color='C0')
    _ax.plot(model_step, losses[model_step], 'o', color='C0')
    _ax.set_xlabel("Training iteration")
    _ax.set_ylabel("Loss")

    mo.accordion({
        "Training progress": _ax
    })

    return


@app.cell
def __(mo, weights_ui):
    mo.md(
        rf"""
        ### Weights
        {weights_ui}
        ### Generated lyrics
        """
    )
    return


@app.cell
def __(
    U,
    corpus_ids,
    generation_seed,
    generation_seed_selector,
    itertools,
    mo,
    model,
    tokenizer,
    vocabulary,
):
    _max_len = len(corpus_ids)
    _gen_ids = itertools.islice(model.generate(corpus_ids[0], seed=generation_seed), _max_len)
    generated = [vocabulary[id] for id in _gen_ids]
    generated = U.tokens_out(generated, tokenizer)

    mo.vstack((generated, generation_seed_selector))
    return generated,


if __name__ == "__main__":
    app.run()
