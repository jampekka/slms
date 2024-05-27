import marimo

__generated_with = "0.6.8"
app = marimo.App(app_title="SLMs basics")


@app.cell
def __():
    import marimo as mo
    from pprint import pformat
    from collections import defaultdict
    def python_out(code):
        #return code
        return mo.Html("<pre>" + pformat(code, sort_dicts=False, compact=True) + "</pre>")
    return defaultdict, mo, pformat, python_out


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
def __(corpus_text, python_out):
    corpus_words = corpus_text.split(' ')
    python_out(corpus_words)
    return corpus_words,


@app.cell
def __(mo):
    mo.md(
        rf"""
        (The here `'\n'` means that we start a new line. While not really a word, we treat it as such for now.)

        We can also build our **vocabulary**, which is just all individual words that is in our lyrics:
        """
    )
    return


@app.cell
def __(corpus_words, python_out):
    # Using dict instead of set to keep the order
    _vocabulary = {w: None for w in corpus_words}.keys()
    python_out(list(_vocabulary))
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The currently popular large language models (LLMs) -- such as GPT, Llama and Mistral -- are based on predicting what token becomes after
        some number of tokens.

        In our case, for example, the word `'Happy'` is followerd by the word `'birthday'` and the
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

    def plot_follower_graph(next_words):
        # TODO: This is fugly. Use dot
        next_words = {repr(k): list(map(repr, v)) for k, v in next_words.items()}
        graph = nx.from_dict_of_lists(next_words, create_using=nx.DiGraph)
        #nx.draw_pydot(graph)
        nx.draw(graph, arrows=True, with_labels=True)
        return plt.gca()

    plot_follower_graph(next_words)
    return nx, plot_follower_graph, plt


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
    initial_lyrics_birthday = tuple(corpus_words[:2])
    get_lyrics_birthday, set_lyrics_birthday = mo.state(initial_lyrics_birthday, allow_self_loops=True)
    return get_lyrics_birthday, initial_lyrics_birthday, set_lyrics_birthday


@app.cell
def __(mo):
    def dropdown_generate(next_words, lyrics_state, initial_lyrics):
        get_lyrics, set_lyrics = lyrics_state
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

        #lyrics_el = mo.Html(f"<pre>{lyrics_text} {dropdown}</pre>")
        return dropdown, reset
    return dropdown_generate,


@app.cell
def __(
    dropdown_generate,
    get_lyrics_birthday,
    initial_lyrics_birthday,
    mo,
    next_words,
    set_lyrics_birthday,
):
    # These have to be globals for the events to be triggered.
    # Marimo has some ways to go to enable modular code
    dropdown_birthday, reset_birthday = dropdown_generate(next_words, (get_lyrics_birthday, set_lyrics_birthday), initial_lyrics_birthday)
    _text = ' '.join(get_lyrics_birthday())
    _lyrics_el = mo.Html(f"<pre>{_text} {dropdown_birthday}</pre>")

    mo.hstack([_lyrics_el, reset_birthday])
    return dropdown_birthday, reset_birthday


@app.cell
def __(mo):
    mo.md(
        rf"""
        ---




        # More context

        The previous looked only one word at the time. However, we can easily use more than one word to predict the next one. How many words (or tokens) we use to predict the next one, is known as the **context length**. The context length of the previous example was 1.

        With the very simple lyrics context length more than 1 does not make much sense, so let's pick something a bit more complicated:
        """
    )
    return


@app.cell
def __():
    blowin_text = """
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
    blowin_text
    return blowin_text,


@app.cell
def __(mo):
    mo.md(
        rf"""
        You may recognize the lyrics. They're the verses of the Bob Dylan's [Blowin' in the Wind](https://www.youtube.com/watch?v=MMFj8uDubsE).

        We proceed like before, first splitting the lyrics into words:
        """
    )
    return


@app.cell
def __(blowin_text, python_out):
    blowin_words = blowin_text.split(' ')
    python_out(blowin_words)
    return blowin_words,


@app.cell
def __(mo):
    mo.md(
        rf"""
        Note that we now have punctuation included in the ''words'', like the comma in `'Yes,'` the question mark in `'man?'`. We also treat two newlines `'\n\n'` as one ''word''. This comes handy, as it separates the verses.

        We now have quite a bit larger vocabulary:
        """
    )
    return


@app.cell
def __(blowin_words, python_out):
    # Using dict instead of set to keep the order
    _vocabulary = {w: None for w in blowin_words}.keys()
    python_out(list(_vocabulary))
    return


@app.cell
def __(mo):
    mo.md(rf"Let's first do the same simple context length 1 model for the new lyrics:")
    return


@app.cell
def __(blowin_words, defaultdict, python_out):
    # Doing this more succintly now
    def get_ngrams(tokens, n):
        for i in range(len(tokens) - n + 1):
            yield tokens[i:i+n]

    blowin_next_words1 = defaultdict(list) 
    for _word, _next_word in get_ngrams(blowin_words, 2):
        blowin_next_words1[_word].append(_next_word)

    python_out(dict(blowin_next_words1))
    return blowin_next_words1, get_ngrams


@app.cell
def __(blowin_next_words1, plot_follower_graph):
    plot_follower_graph(blowin_next_words1)
    return


@app.cell
def __(mo):
    mo.md(rf"We can now generate some lyrics with the model. Here's some machine generated ones, you can do your own below.")
    return


@app.cell
def __():
    import random
    random.seed(3)
    return random,


@app.cell
def __(mo):
    regen_blowin1_btn = mo.ui.button(label="Generate new lyrics")
    regen_blowin1_btn
    return regen_blowin1_btn,


@app.cell
def genblow1_1(blowin_next_words1, random, regen_blowin1_btn):
    regen_blowin1_btn

    def _generate(next_words):
        context = next(iter(next_words.keys()))
        yield context

        while True:
            choices = next_words[context]
            next_word = random.choice(choices)
            if next_word == '\n\n': return
            yield next_word
            context = next_word

    _generated = list(_generate(blowin_next_words1))
    ' '.join(_generated)
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        Don't seem to make much sense. But remember that this has to guess the next word just based on the previous one.

        Try it out yourself! This time the generated lyrics are hidden. Don't peek at them before you're done, and pretend you don't remember what you picked before!
        """
    )
    return


@app.cell
def __(blowin_words, mo):
    initial_lyrics_blowin = blowin_words[:2]
    get_lyrics_blowin1, set_lyrics_blowin1 = mo.state(initial_lyrics_blowin, allow_self_loops=True)
    return get_lyrics_blowin1, initial_lyrics_blowin, set_lyrics_blowin1


@app.cell
def __(
    blowin_next_words1,
    dropdown_generate,
    get_lyrics_blowin1,
    initial_lyrics_blowin,
    mo,
    set_lyrics_blowin1,
):
    dropdown_blowin1, reset_blowin1 = dropdown_generate(blowin_next_words1, (get_lyrics_blowin1, set_lyrics_blowin1), initial_lyrics_blowin)
    _lyrics_el = mo.Html(f"<pre>{repr(get_lyrics_blowin1()[-1])} {dropdown_blowin1}</pre>")

    _lyrics_el
    return dropdown_blowin1, reset_blowin1


@app.cell
def __(get_lyrics_blowin1, mo, reset_blowin1):
    _lyrics = ' '.join(get_lyrics_blowin1())
    _spoiler = mo.accordion({'SPOILER': mo.Html(f"<pre>{_lyrics}</pre>")})
    mo.hstack([_spoiler, reset_blowin1])
    return


if __name__ == "__main__":
    app.run()
