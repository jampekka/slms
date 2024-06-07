import marimo

__generated_with = "0.6.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import utils as U
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F

    np.set_printoptions(precision=2)
    return F, U, mo, nn, np, pd, plt, torch


@app.cell
def __():
    # Define options

    _happy_birthday_text = """
     Happy birthday to you 
     Happy birthday to you 
     Happy birthday dear Dave 
     Happy birthday to you 
    """

    _blowin_text_full = """
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
        "Blowin' in the wind": _blowin_text,
        "Blowin' (all verses)": _blowin_text_full
    }

    n_iterations = 60
    context_length = 1
    return context_length, corpus_selections, n_iterations


@app.cell
def __(corpus_selections, mo, n_iterations):
    # Define controls
    corpus_selector = mo.ui.dropdown(corpus_selections, value="Happy Birthday")
    iter_selector = mo.ui.slider(0, n_iterations, value=20,
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
    vocab_labels = list(map(repr, vocabulary))
    return (
        corpus_words,
        tokenizer,
        vocab_labels,
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

        Let's analyze lyrics of {corpus_name}. We use the simple word tokenizer to keep things more intuitive. The neural network works exactly the same with other tokenizers. See the [Tokenization](?file=tokenization.py) notebook for further info about tokenizers.
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

        And let's do the simple context length one model, where we predict the next word just based on the current word like we did in [Basics](?file=basics.py).
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
    vocab_labels,
    vocabulary,
    vocabulary_size,
):
    import itertools

    _current_to_next = list(itertools.permutations(range(vocabulary_size, 2)))

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
        # TODO: This doesn't seem to update properly
        df = pd.DataFrame(connections,
                          columns=left_labels,
                          index=left_labels
        )
        df = df.round(2) # Can't seem to control print precision otherwise
        return df
        #return mo.ui.table(df, selection=None)

    _plot = plot_nn_layer(_connections, vocab_labels, vocab_labels)
    _table = nn_weights_table(_connections, vocab_labels, vocab_labels)
    mo.ui.tabs({
        "Weight graph": _plot,
        "Weight table": _table
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
            torch.manual_seed(0)
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
    from copy import deepcopy

    corpus_ids = [word_vocab_pos[w] for w in corpus_words]

    dataset = torch.as_tensor(list(U.get_ngrams(corpus_ids, context_length+1)))
    contexts = dataset[:,:-1]
    targets = dataset[:,-1]

    _model = SimpleNnLm(vocabulary_size)

    def train_model(model):
        torch.manual_seed(0)

        #with torch.no_grad():
        #    for p in model.parameters():
        #        p *= 0.0
        #        p += 1.0
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e1)
        
        losses = []
        model_steps = []
        for _i in range(n_iterations + 1):
            predictions = model(contexts).squeeze(1)
            loss = criterion(predictions, targets.view(-1))
            # Add small L1 regularization to the weights to
            # get more sparse ones for nicer visualization.
            for p in model.parameters():
                loss += torch.norm(p.flatten()*1e-3, 1)
            model_steps.append(deepcopy(model))
            losses.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses, model_steps

    losses, model_steps = train_model(_model)
    return (
        contexts,
        corpus_ids,
        dataset,
        deepcopy,
        losses,
        model_steps,
        targets,
        train_model,
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
def __(losses, mo, model_step, plt):
    def plot_training_process(losses, color='C0', label=None, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(range(len(losses)), losses, color=color, label=label)
        ax.plot(model_step, losses[model_step], 'o', color=color)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss")
        return ax

    mo.accordion({
        "Training progress": plot_training_process(losses)
    })
    return plot_training_process,


@app.cell
def __(mo, weights_ui):
    mo.md(rf"{weights_ui}")
    return


@app.cell
def __(mo):
    mo.md(rf"### Generated lyrics")
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


@app.cell
def __(F, nn, torch):
    class SimpleEmbedNnLm(nn.Module):
        def __init__(self, vocab_size, embed_size):
            torch.manual_seed(0)
            super().__init__()
            self.vocab_size = vocab_size
            self.embed_size = embed_size
            self.embedder = nn.Embedding(vocab_size, embed_size)
            # Norming to keep the weights in a nice scale
            #self.normer = nn.LayerNorm(embed_size, elementwise_affine=False, bias=False)
            self.normer = nn.LayerNorm(embed_size)
            self.mlp = nn.Linear(embed_size, vocab_size, bias=False)

        def forward(self, input):
            embedding = self.embedder(input)
            #embedding = self.normer(embedding)
            #logits = F.relu(self.mlp(embedding))
            logits = self.mlp(embedding)
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
    return SimpleEmbedNnLm,


@app.cell
def __(mo):
    #embed_size_selector = mo.ui.slider(1, 10, value=2, label="Embedding size", full_width=True, show_value=True)
    _embed_choices = {str(i): i for i in range(1, 11)}
    embed_size_selector = mo.ui.dropdown(_embed_choices, value='2')

    return embed_size_selector,


@app.cell
def __(SimpleEmbedNnLm, embed_size_selector, train_model, vocabulary_size):
    embed_size = embed_size_selector.value

    _model = SimpleEmbedNnLm(vocabulary_size, embed_size)
    #_model2 = SimpleNnLm(vocabulary_size)

    losses_e, model_steps_e = train_model(_model)
    return embed_size, losses_e, model_steps_e


@app.cell
def __(model_step, model_steps_e):
    model_e = model_steps_e[model_step]
    return model_e,


@app.cell
def __(
    corpus_selector,
    corpus_words,
    embed_size,
    embed_size_selector,
    mo,
    vocabulary_size,
):
    mo.md(
        rf"""
        ## Embeddings

        Currently our neural network is actually rather large.

        In {corpus_selector} we have `{vocabulary_size}` different tokens (words). To predict from each word to each word, we need `{vocabulary_size}x{vocabulary_size} = {vocabulary_size*vocabulary_size}` weights in our neural network. Given that we have a training set of only {len(corpus_words)} tokens, we are sure to overfit the data, i.e. just memorize the lyrics that we have.

        But we can quite easily make it smaller by adding a smaller **hidden layer** often called an **embedding layer**. Let's try an embedding layer size {embed_size_selector}. 

        This means we will first have `{vocabulary_size}x{embed_size}` input weights and the same amount of output weights. In total we'll have `{vocabulary_size}x{embed_size} + {embed_size}x{vocabulary_size} = {2*embed_size*vocabulary_size}` weights. That's about {round(2*embed_size*vocabulary_size/vocabulary_size**2*100)}% of the original weights!

        Try out the different lyrics and different embedding sizes. Typically for more complicated lyrics we need more embedding layers to fit the data.
        """
    )
    return


@app.cell
def __(iter_selector):
    iter_selector
    return


@app.cell
def __(losses, losses_e, mo, plot_training_process, plt):
    _, _ax = plt.subplots()

    plot_training_process(losses, label="No embedding", color='C0', ax=_ax)
    plot_training_process(losses_e, label="With embedding", color='C1', ax=_ax)
    _ax.legend()
    mo.accordion({
        "Training progress": _ax
    })
    return


@app.cell
def __(embed_size, mo, vocabulary_size):
    mo.md(rf"Now we have weights from the original `{vocabulary_size}` words to the `{embed_size}` hidden units. Note that the weights can be now also negative. Positive weights are plotted in blue and negative ones in red.")
    return


@app.cell
def __(F, model_e, np, plt, vocab_labels):
    def plot_nn(layers, left_labels, right_labels):
        # TODO: Line colors
        from matplotlib.collections import LineCollection
        #froms, tos = np.nonzero(connections)
        #w = connections[x, y]
        ax = plt.gca()
        tax = ax.twinx()
        for i, connections in enumerate(layers):
            leftpos = i
            rightpos = leftpos + 1
            segs = []
            alphas = []
            colors = []

            nx, ny = connections.shape

            for x in range(nx):
                for y in range(ny):
                    weight = connections[x, y]
                    xc = (x + 0.5)/nx - 0.5
                    yc = (y + 0.5)/ny - 0.5
                    seg = (leftpos, xc), (rightpos, yc)
                    segs.append(seg)

                    color = ['blue', 'red'][weight < 0]
                    alpha = np.minimum(1, np.abs(weight))*0.8
                    alphas.append(alpha)
                    colors.append(color)
            
            """
            nx, ny = connections.shape
            #xs, ys = np.indices((nx, ny), sparse=True)
            xs, ys = map(np.ravel, np.mgrid[0:nx, 0:ny])
            weights = connections[xs, ys]

            token_positions = np.vstack((xs, ys)).T
            n = len(token_positions)
            side_positions = np.vstack((np.repeat(leftpos, n), np.repeat(rightpos, n))).T
            segs = np.dstack((side_positions, token_positions))
            alpha = np.minimum(1, np.abs(weights))*0.8
            """
            lc = LineCollection(segs, alpha=alphas, color=colors)
            
            ax.add_collection(lc)
            
        for a in [ax, tax]:
            ticks = (np.arange(ny) + 0.5)/ny - 0.5
            a.set_yticks(ticks=ticks, labels=left_labels)
            a.set_ylim(ticks[-1], ticks[0])
            a.set_xlim(0, len(layers))
            for spine in a.spines.values():
                spine.set_visible(False)

        ax.set_xticks([])

        return ax

    def _mangle_weights(w):
        return F.tanh(w*0.1).detach()

    plot_nn((
        _mangle_weights(model_e.embedder.weight),
        _mangle_weights(model_e.mlp.weight).T
           ), vocab_labels, vocab_labels)
            
    return plot_nn,


@app.cell
def __(
    U,
    corpus_ids,
    generation_seed,
    generation_seed_selector,
    itertools,
    mo,
    model_e,
    tokenizer,
    vocabulary,
):
    _max_len = len(corpus_ids)
    _gen_ids = itertools.islice(model_e.generate(corpus_ids[0], seed=generation_seed), _max_len)
    _generated = [vocabulary[id] for id in _gen_ids]
    _generated = U.tokens_out(_generated, tokenizer)

    mo.vstack((_generated, generation_seed_selector))
    return


@app.cell
def __(corpus_selector, embed_size_selector, mo):
    mo.md(
        rf"""
        For each word we can compute its embedding by computing what embedding value it ends up to.

        As embeddings are numbers, they can be also plotted on x-y axis. If the dimension is more than `2`, we compute Principal Axis Decomposition and use the first two component. Below we plot the 2 dimensional projection of the {embed_size_selector} dimensional embeddings for {corpus_selector}.

        With larger datasets and networks the embedding values tend to exhibit some semantic-looking behavior. In general, words that occur in similar context tend to have similar embedding values.
        """
    )
    return


@app.cell
def __(model_e, np, plt, torch, vocab_labels):
    def plot_embeddings(weights):
        torch.manual_seed(0)

        #print(weights.shape)
        if weights.shape[1] == 1:
            comps = np.vstack((
                np.zeros(weights.shape[0]),
                weights[:,0])).T
        elif weights.shape[1] == 2:
            comps = weights
        else:
            U, S, V = torch.pca_lowrank(weights)
            comps = (weights@V[:,:2]).detach()
        plt.plot(*comps.T, 'k.')
        for i, (x, y) in enumerate(comps):
            plt.text(x, y, vocab_labels[i])
        plt.xlabel("Embedding component 1")
        plt.ylabel("Embedding component 2")
        return plt.gca()
    plot_embeddings(model_e.embedder.weight.detach())
    return plot_embeddings,


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## Going deeper

        In these examples we used only context length of `1` and a neural network with just one hidden layer. The purpose was to give a conceptual understanding of neural network models, and conceptually this simple one has most of the components of large language models.

        If you want to study the more complicated models, I can recommend the video series by [3blue1brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) for more conceptual understanding, and if you want to understand how the models are programmed, see [Andrew Karpathy's video series](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).
        """
    )
    return


if __name__ == "__main__":
    app.run()
