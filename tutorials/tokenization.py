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


        Let's get back to the Happy Birthday lyrics. We will later study other texts too.
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
        rf"""
        We'll start with the same word tokenizer now. We also switched from listing all of the following tokens by storing a count of how many times they occur after a context. This does the exact same thing, but in a bit nicer format.

        If ou click on the different accordion tabs below, you should see familiar looking results.
        """
    )
    return


@app.cell
def __(mo):
    context_length_slider = mo.ui.slider(start=1, value = 2, stop=8, full_width=True)
    return context_length_slider,


@app.cell
def __(context_length_slider, mo):
    mo.md(
        f"""
        The context length is {context_length_slider.value}

        {context_length_slider}
        """
    )
    return


@app.cell
def __(U, context_length_slider, corpus_text, mo, s):
    class WordTokenizer:
        def __call__(self, s):
            return s.split(' ')
        def detokenize(self, tokens):
            return ' '.join(tokens)

    class CharacterTokenizer:
        def __call__(self, s):
            return list(s)

        def detokenize(self, tokens):
            return ''.join(s)

    tokenizers = {
        "word": WordTokenizer()
    }

    tokenizer_type = "word"
    tokenizer = tokenizers[tokenizer_type]
    context_length = context_length_slider.value

    corpus_tokens = tokenizer(corpus_text)
    vocabulary = tokenizer(corpus_text)
    next_tokens = U.get_next_token_table(corpus_tokens, context_length)

    mo.accordion({
        "Tokenized text": U.python_out(corpus_tokens),
        "Vocabulary": U.python_out(vocabulary),
        "Next token table": U.python_out(dict(next_tokens)),
        "Follower graph": U.plot_follower_graph(next_tokens),
    })
    return (
        CharacterTokenizer,
        WordTokenizer,
        context_length,
        corpus_tokens,
        next_tokens,
        tokenizer,
        tokenizer_type,
        tokenizers,
        vocabulary,
    )


if __name__ == "__main__":
    app.run()
