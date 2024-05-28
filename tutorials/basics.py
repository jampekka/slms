import marimo

__generated_with = "0.6.8"
app = marimo.App(app_title="SLMs basics")


@app.cell
def __():
    import marimo as mo
    from pprint import pformat
    from collections import defaultdict
    import utils as U

    U.init_output
    return U, defaultdict, mo, pformat


@app.cell
def __(mo):
    mo.md(
        r"""
        # Small language models

        ## Happy birthday
        ---
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
def __(U, corpus_text):
    corpus_words = corpus_text.split(' ')
    U.python_out(corpus_words)
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
def __(U, corpus_words):
    # Using dict instead of set to keep the order
    _vocabulary = {w: None for w in corpus_words}.keys()
    U.python_out(list(_vocabulary))
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
def __(U, corpus_words):
    next_words = {}
    for i in range(len(corpus_words)-1):
        word = corpus_words[i]
        next_word = corpus_words[i+1]
        if word not in next_words:
            next_words[word] = []
        next_words[word].append(next_word)
    U.python_out(next_words)
    return i, next_word, next_words, word


@app.cell
def __(mo):
    mo.md(r"Or as a visual graph format:")
    return


@app.cell
def __(U, next_words):
    U.plot_follower_graph(next_words)
    return


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
        ## Blowin' in the wind
        ---

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
        You may recognize the lyrics. They're the verses of the Bob Dylan's song [Blowin' in the Wind](https://www.youtube.com/watch?v=MMFj8uDubsE).

        We proceed like before, first splitting the lyrics into words:
        """
    )
    return


@app.cell
def __(U, blowin_text):
    blowin_words = blowin_text.split(' ')
    U.python_out(blowin_words)
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
def __(U, blowin_words):
    U.python_out(list(U.corpus_to_vocabulary(blowin_words)))
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ### More context
        ---
        We build a simple language model again with these lyrics. These simple models are usually called ''Markov Chain text generators''. This is a bit misleading, as even the next-token-predicting LLMs are Markov chains. We won't discuss what Markov chains really are and what makes a model such, but Wikipedia has a [rather good article](https://en.wikipedia.org/wiki/Markov_chain) of these if you're interested. 

        Previously in the ''Happy Birthday'' example the model looked only one word at the time. However, we can easily use more than one word to predict the next one. How many words (or tokens) we use to predict the next one, is known as the **context length**. The context length of the previous example was 1.

        For lyrics as simple as in ''Happy Birthday'' using a context length more than 1 didn't make much sense. However, with the more complicated lyrics we can see how the model behavior changes with different context lengths.

        You can select the context length with the slider and see how the model changes.
        """
    )
    return


@app.cell
def __(context_length_slider, mo):
    mo.md(f"The context length is {context_length_slider.value}")
    return


@app.cell
def __(mo):
    # TODO: Display context length value
    context_length_slider = mo.ui.slider(start=1, stop=8, full_width=True)
    context_length_slider
    return context_length_slider,


@app.cell
def __(blowin_words, context_length_slider, defaultdict):
    #blowin_context_length = 2
    blowin_context_length = context_length_slider.value
    # Doing this more succintly now
    def get_ngrams(tokens, n):
        for i in range(len(tokens) - n + 1):
            yield tokens[i:i+n]

    blowin_next_words1 = defaultdict(list) 
    for *_context, _next_word in get_ngrams(blowin_words, blowin_context_length + 1):
        blowin_next_words1[tuple(_context)].append(_next_word)

    #python_out(dict(blowin_next_words1))
    return blowin_context_length, blowin_next_words1, get_ngrams


@app.cell
def __():
    #plot_follower_graph(blowin_next_words1)
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
    regen_blowin1_btn = mo.ui.button(label="Generate new verse")
    regen_blowin1_btn
    return regen_blowin1_btn,


@app.cell
def genblow1_1(U, blowin_next_words1, random, regen_blowin1_btn):
    # TODO: Keep the seed constant across generations

    regen_blowin1_btn

    def _generate(next_words):
        context = next(iter(next_words.keys()))
        yield from context

        while True:
            choices = next_words[context]
            if not choices: return
            next_word = random.choice(choices)
            if next_word == '\n\n': return
            yield next_word
            context = (*context[1:], next_word)

    _generated = list(_generate(blowin_next_words1))
    U.pre_box(' '.join(_generated))
    return


@app.cell
def __(U, blowin_next_words1, mo):
    mo.accordion({
        "Next word table": U.python_out(dict(blowin_next_words1)),
        "Next word graph": U.plot_follower_graph(blowin_next_words1)
    })
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        With a short context length the lyrics dont make much sense. With a longer context length it starts to just copy the originals. Try to find a context length that seems to make a nice tradeoff between these. As a hint, you can get something quite silly with some context lengths.

        Try to be such a language model yourself! This time the generated lyrics are hidden. Don't peek at them before you're done, and pretend you don't remember what you picked before!
        """
    )
    return


@app.cell
def __(blowin_context_length, blowin_words, mo):
    initial_lyrics_blowin = blowin_words[:blowin_context_length + 1]
    get_lyrics_blowin1, set_lyrics_blowin1 = mo.state(initial_lyrics_blowin, allow_self_loops=True)
    return get_lyrics_blowin1, initial_lyrics_blowin, set_lyrics_blowin1


@app.cell
def __(
    blowin_context_length,
    blowin_next_words1,
    get_lyrics_blowin1,
    initial_lyrics_blowin,
    mo,
    set_lyrics_blowin1,
):
    def dropdown_generate_blowin(next_words, lyrics_state, initial_lyrics):
        get_lyrics, set_lyrics = lyrics_state
        lyrics = get_lyrics()
        context = tuple(lyrics[-blowin_context_length:])
        options = set(next_words[context])
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

    dropdown_blowin1, reset_blowin1 = dropdown_generate_blowin(blowin_next_words1, (get_lyrics_blowin1, set_lyrics_blowin1), initial_lyrics_blowin)
    _ctx = ', '.join(map(repr, get_lyrics_blowin1()[-blowin_context_length:]))
    _lyrics_el = mo.Html(f"<pre>{_ctx} {dropdown_blowin1}</pre>")

    _lyrics_el
    return dropdown_blowin1, dropdown_generate_blowin, reset_blowin1


@app.cell
def __(get_lyrics_blowin1, mo, reset_blowin1):
    _lyrics = ' '.join(get_lyrics_blowin1())
    _spoiler = mo.accordion({'Your generated lyrics. SPOILER!': mo.Html(f"<pre>{_lyrics}</pre>")})
    mo.vstack([_spoiler, reset_blowin1])
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ---
        In the next notebook, we'll take a closer look at **tokenization**, i.e. how we split the text for processing.

        [Continue to Tokenization >](?file=tokenization.py)
        """
    )
    return


if __name__ == "__main__":
    app.run()
