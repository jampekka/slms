import marimo

__generated_with = "0.6.8"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    import marimo as mo
    from pprint import pformat
    from collections import defaultdict
    import utils as U

    U.init_output
    return U, defaultdict, mo, pformat


@app.cell
def __(mo):
    mo.md(
        rf"""
        # Tokenization
        ---
        In the [previous notebook](?file=basics.py) we split the text roughly to words. But the models don't care about what the pieces of text are, and we can split them any way we want. In this notebook we can try out different tokenizations and see how they affect the model's behavior.

        Now the text will separate tokens with alternate colored backgrounds, so they can be distinguished easier. We also print out the newline characters and show spaces as underlined.

        Last time we also did a trick where spaces `' '` were not tokens. In the following we'll treat them as separate tokens too.

        Now you should see some familiar lyrics tokenized like this:
        """
    )
    return


@app.cell
def __(U):
    class Tokenizer:
        def tokens_to_strings(self, tokens):
            return map(self.token_to_string, tokens)

        def detokenize(self, tokens):
            strings = self.tokens_to_strings(tokens)
            return ''.join(strings)

        def token_to_string(self, s):
            return s

    class HackyWordTokenizer(Tokenizer):
        def __call__(self, s):
            return s.split(' ')

        def tokens_to_strings(self, tokens):
            for token in tokens:
                yield token
                # TODO: Shouldn't yield last space
                yield ' '

    import re
    class WordTokenizer(Tokenizer):
        def __call__(self, s):
            out = re.split('( +|\n+)', s)
            return [t for t in out if t]

    class CharacterTokenizer(Tokenizer):
        def __call__(self, s):
            return list(s)

    import transformers

    #_BASE_MODEL="EleutherAI/pythia-14m"
    _BASE_MODEL="facebook/opt-125m"
    class SubwordTokenizer(Tokenizer):
        def __init__(self):
            self._tok = transformers.AutoTokenizer.from_pretrained(_BASE_MODEL)

        def __call__(self, s):
            # Using strings instead of ids to avoid confusion
            token_ids = self._tok(s)['input_ids']
            return [self._tok.decode([id]) for id in token_ids]


    tokenizers = {
        "Word": WordTokenizer(),
        "Character": CharacterTokenizer(),
        "Subword": SubwordTokenizer(),
    }

    languages = {
        "English": U.blowin_text,
        "Finnish": U.blowin_text_finnish,
        "German": U.blowin_text_german,
    }


    return (
        CharacterTokenizer,
        HackyWordTokenizer,
        SubwordTokenizer,
        Tokenizer,
        WordTokenizer,
        languages,
        re,
        tokenizers,
        transformers,
    )


@app.cell
def __(languages, mo):
    language_selector = mo.ui.dropdown(options=languages, value="English", allow_select_none=False)

    random_seed_slider = mo.ui.slider(start=1, value=1, stop=30, full_width=False, show_value=True, label="Variation (random seed)")
    return language_selector, random_seed_slider


@app.cell
def __(language_selector):
    corpus_text = language_selector.value
    return corpus_text,


@app.cell
def __(corpus_text, language_selector, mo, tokenizers):
    corpus_text_first_line = corpus_text.strip().split('\n')[0]

    tokenizer_texts = {
        "Word tokenizer": mo.md("""
        The word tokenizer splits the text into individual words. This tends to generate somewhat legible text even with short context lengths. However, it can't create new words!

        This is not so bad in English language that has very few inflections. However, in synthetic and agglutinative languages like Finnish this is a big problem, as you can form new words that have never been uttered in the history of the world!
        """),
        "Character tokenizer": mo.md("""
        The character tokenizer splits the text into individual characters. With this we can create new words, but especially with shorter context length, it produces total gibberish!

        A tradeoff between word tokenization and character tokenization is **subword tokenization**. Here common strings, like English words and Finnish inflections, are typically represented as a single token, but the tokenization also includes individual characters.
        """),
        "Subword tokenizer": mo.md(f"""
        Subword tokenizer tries to split the text to commonly occuring strings, such as words, but it can "fall back" to smaller strings, including single characters. Typically most common English words are individual tokens. Also subwords like Finnish inflections or syllables may get their own tokens.

        A common method for subword tokenization is [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding). The tokenizer in this examples uses that method, and is in fact the same tokenizer that was used for GPT-2.

        You may notice that for English the resulting tokenization is not that different from the word tokenization. A major difference is that spaces are included in the tokens. However, see what happens if you do a Finnish or German translation of the lyrics: {language_selector}
        """),
    }

    context_length_slider = mo.ui.slider(start=1, value = 2, stop=10, full_width=False, label="Context length", show_value=True)

    tokenizer_tabs = mo.ui.tabs(
        tokenizer_texts
    )
    tokenizer_selector = mo.ui.dropdown(options=tokenizers.keys(), value="Word", label="Tokenizer")
    return (
        context_length_slider,
        corpus_text_first_line,
        tokenizer_selector,
        tokenizer_tabs,
        tokenizer_texts,
    )


@app.cell
def __(U, context_length_slider, corpus_text, tokenizer_tabs, tokenizers):
    tokenizer_type = tokenizer_tabs.value.split()[0]
    tokenizer = tokenizers[tokenizer_type]
    context_length = context_length_slider.value

    corpus_tokens = tokenizer(corpus_text)
    vocabulary = U.corpus_to_vocabulary(corpus_tokens)
    next_tokens = U.get_next_token_table(corpus_tokens, context_length)


    U.tokens_out(corpus_tokens, tokenizer)
    return (
        context_length,
        corpus_tokens,
        next_tokens,
        tokenizer,
        tokenizer_type,
        vocabulary,
    )


@app.cell
def __(mo):
    mo.md(rf"With the tabs below, you can select different tokenizers. As you change the tokenizer, the results below change automatically. Go through the different tokenizers and observe how they change the results!")
    return


@app.cell
def __(mo, tokenizer_tabs):
    mo.md(
        f"""
        ## Tokenizer selection
        ---
        <div style="height: 20em; overflow: auto;">
        {tokenizer_tabs}
        </div>
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        f"""
        ## Playground (watch this change!)
        ---
        """
    )
    return


@app.cell
def __(
    U,
    context_length_slider,
    corpus_tokens,
    mo,
    next_tokens,
    random_seed_slider,
    tokenizer,
):
    gen_seed = random_seed_slider.value
    gen_tokens = U.generate_tokens(next_tokens, seed=gen_seed)

    gen_ui = mo.vstack([
        U.tokens_out(gen_tokens, tokenizer),
        mo.hstack([context_length_slider, random_seed_slider])
    ])

    mo.ui.tabs({
        "Random generated": gen_ui,
        "Tokenized original": U.tokens_out(corpus_tokens, tokenizer),
        #"Follower graph": U.plot_follower_context_graph(next_tokens),
        #"Vocabulary": U.python_out(vocabulary),
        #"Next token table": U.python_out(dict(next_tokens)),
    })
    return gen_seed, gen_tokens, gen_ui


@app.cell
def __(language_selector, mo):
    mo.md(rf"Lyrics language {language_selector} (Translation by Google Translate)")
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ---
        In the next notebook, we'll learn basics of neural networks and how they can be used to create more flexible and scalable language models.

        [Continue to Neural Networks >](?file=tokenization.py)
        """
    )
    return


if __name__ == "__main__":
    app.run()
