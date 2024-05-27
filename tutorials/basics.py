import marimo

__generated_with = "0.6.8"
app = marimo.App(app_title="SLMs basics")


@app.cell
def __():
    import marimo as mo
    from pprint import pformat
    def python_out(code):
        return mo.md("```" + repr(code) + "```")
    return mo, pformat, python_out


@app.cell
def __(mo):
    mo.md(
        r"""
        # The very basics of language models

        To get started, we analyze lyrics of perhaps the most popular song in the world.
        You may be familiar with the lyrics:
        """
    )
    return


@app.cell
def __():
    corpus_text = """
     Happy birthday to you 
     Happy birthday to you 
     Happy birthday dear Dave 
     Happy birthday to you 
    """
    corpus_text
    return corpus_text,


@app.cell
def __(mo):
    mo.md(
        r"""
        To work with text, we usually want to split it to some shorter pieces, such
        as words. In general, such pieces are called **tokens**, but we'll start with just
        words. Our lyrics split into words become:
        """
    )
    return


@app.cell
def __(corpus_text):
    corpus_words = corpus_text.split(' ')
    corpus_words
    return corpus_words,


@app.cell
def __(mo):
    mo.md(
        r"""
        (The here `'\n'` means that we start a new line. While not really a word, we treat it as such for now.)

        The currently popular large language models (LLMs) are based on predicting what token becomes after
        some number of tokens. For example, the word `'Happy'` is followerd by the word `'birthday'` and the
        word `'birthday'` is followed by the word `'to'`.

        In fact, to make an extremely simple language model, we can just list what words are followed by each
        word. For our lyrics this becomes:
        """
    )
    return


@app.cell
def __(corpus_words, python_out):
    next_words = {}
    for i in range(len(corpus_words)-1):
        word = corpus_words[i]
        next_word = corpus_words[i+1]
        if word not in next_words:
            next_words[word] = []
        next_words[word].append(next_word)
    python_out(next_words)
    return i, next_word, next_words, word


@app.cell
def __(mo):
    mo.md(r"Or as a visual graph format:")
    return


@app.cell
def __(next_words):
    import networkx as nx
    import matplotlib.pyplot as plt
    graph = nx.from_dict_of_lists(next_words, create_using=nx.DiGraph)
    nx.draw_circular(graph, arrows=True, with_labels=True)
    plt.gca()
    return graph, nx, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see that after a new line `'\n'` we always get the word `'Happy'`, and `'Happy'` is always followed by
        `'birthday'`. Somewhat more interestingly, the word `'birthday'` was followed three times by `'to'` but also
        once by `'dear'`.

        With this model, we are ready to generate new lyrics! Select the next word from the dropdown
        to add it into the lyrics.
        """
    )
    return


@app.cell
def __(corpus_words, mo):
    initial_lyrics = tuple(corpus_words[:2])
    get_lyrics, set_lyrics = mo.state(initial_lyrics, allow_self_loops=True)
    return get_lyrics, initial_lyrics, set_lyrics


@app.cell
def __(get_lyrics, initial_lyrics, mo, next_words, set_lyrics):
    lyrics = get_lyrics()
    options = set(next_words[lyrics[-1]])
    def update(value):
        new_lyrics = (*get_lyrics(), value)
        set_lyrics((*get_lyrics(), value))

    lyrics_text = ' ' + ' '.join(get_lyrics())

    optvals = {repr(o): o for o in options}
    dropdown = mo.ui.dropdown(options=optvals, on_change=update)
    reset = mo.ui.button(
        label="Reset lyrics",
        on_change=lambda *args: set_lyrics(initial_lyrics)
    )

    lyrics_el = mo.Html(f"<pre>{lyrics_text} {dropdown}</pre>")

    mo.hstack([lyrics_el, reset])
    return (
        dropdown,
        lyrics,
        lyrics_el,
        lyrics_text,
        options,
        optvals,
        reset,
        update,
    )


if __name__ == "__main__":
    app.run()
