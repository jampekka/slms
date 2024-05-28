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


        Let's get back to the Happy Birthday lyrics. We will later study other texts too. Now the text will separate tokens with different colors, so they can be distinguished easier. We also print out the newline characters and show spaces as underlined.

        Last time we also did a trick where spaces `' '` were not tokens. In the following we'll treat them as separate tokens too.
        """
    )
    return


@app.cell
def __(mo):
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


    tokenizers = {
        "Word": WordTokenizer(),
        "Character": CharacterTokenizer()
    }

    context_length_slider = mo.ui.slider(start=1, value = 2, stop=10, full_width=True)
    tokenizer_selector = mo.ui.dropdown(options=tokenizers.keys(), value="Word", label="Tokenizer")
    random_seed_slider = mo.ui.slider(start=1, value=1, stop=30, full_width=True, label="Variation (random seed)")





    return (
        CharacterTokenizer,
        HackyWordTokenizer,
        Tokenizer,
        WordTokenizer,
        context_length_slider,
        random_seed_slider,
        re,
        tokenizer_selector,
        tokenizers,
    )


@app.cell
def __(U, context_length_slider, tokenizer_selector, tokenizers):
    corpus_text = U.blowin_text

    tokenizer_type = tokenizer_selector.value
    tokenizer = tokenizers[tokenizer_type]
    context_length = context_length_slider.value

    corpus_tokens = tokenizer(corpus_text)
    vocabulary = U.corpus_to_vocabulary(corpus_tokens)
    next_tokens = U.get_next_token_table(corpus_tokens, context_length)


    U.tokens_out(corpus_tokens, tokenizer)
    return (
        context_length,
        corpus_text,
        corpus_tokens,
        next_tokens,
        tokenizer,
        tokenizer_type,
        vocabulary,
    )


@app.cell
def __(mo):
    mo.md(
        rf"""
        We'll start with the same word tokenizer now. We also switched from listing all of the following tokens by storing a count of how many times they occur after a context. This does the exact same thing, but in a bit nicer format.

        If ou click on the different accordion tabs below, you should see familiar looking results.
        """
    )
    return


@app.cell
def __(context_length_slider, mo, tokenizer_selector):
    mo.md(
        f"""
        The context length is {context_length_slider.value}

        {context_length_slider}

        {tokenizer_selector}
        """
    )
    return


@app.cell
def __(
    U,
    corpus_tokens,
    mo,
    next_tokens,
    random_seed_slider,
    tokenizer,
    vocabulary,
):
    gen_seed = random_seed_slider.value
    gen_tokens = U.generate_tokens(next_tokens, seed=gen_seed)

    gen_ui = mo.vstack([
        U.tokens_out(gen_tokens, tokenizer),
        random_seed_slider
    ])

    mo.ui.tabs({
        "Random generated": gen_ui,
        "Tokenized text": U.python_out(corpus_tokens),
        #"Follower graph": U.plot_follower_context_graph(next_tokens),
        "Vocabulary": U.python_out(vocabulary),
        "Next token table": U.python_out(dict(next_tokens)),
    })
    return gen_seed, gen_tokens, gen_ui


if __name__ == "__main__":
    app.run()
