import marimo as mo

from pprint import pformat
from collections import defaultdict, Counter
import random

_pre_box_height = "10em";
_font_size = "12px";
def pre_box(text):
    return mo.Html(f"""
<pre class="pre_out_box" style="overflow: auto; height: {_pre_box_height}; font-size: {_font_size};">
{text}

</pre>""")

def python_out(code):
    return mo.Html(f"""
<pre class="python_out_box" style="overflow: auto; height: {_pre_box_height}; font-size: {_font_size};">
{pformat(code, sort_dicts=False, compact=True)}

</pre>""")

def tokens_out(tokens, tokenizer):
    out = ""
    for i, string in enumerate(tokenizer.tokens_to_strings(tokens)):
        #colors = ["rgb(20, 184, 166)", "rgb(245, 158, 11)"]
        colors = [
                "#2b9a66",
                #"#26997b",
                "#00749e",
                "#dc3e42",
        ]
        #colors = "#d1f0fa", "#ffcdce"
        colors = "var(--sky-3)", "var(--red-3)", "var(--amber-3)"
        color = colors[i%len(colors)]
        # TODO: Be more general!
        if string == ' ':
            decoration = "underline"
        else:
            decoration = "none"

        n_newlines = string.count('\n')
        string = string.replace("\n", "\\n")
        string += "\n"*n_newlines

        out += f'<span style="background-color: {color}; text-decoration: {decoration}">{string}</span>'
    #out = f'<div style="overflow: auto; height: {_pre_box_height};">{out}</div>'
    return pre_box(out)

def corpus_to_vocabulary(tokens):
    # Using dict instead of set to keep the order
    return list({w: None for w in tokens}.keys())

init_output = mo.Html(f"""
    <style>
    .python_out_box {{
        overflow: auto !important;
        max_height: {_pre_box_height};
        font-size: 12px;
    }}

    .pre_out_box {{
        overflow: auto !important;
        height: {_pre_box_height};
        font-size: 12px;
    }}
    </style>
    """)
init_output = None

def graph_out(svg):
    return mo.Html(f"""
        <div style="overflow: auto; max-height: 32em;">
        {svg}
        </div>
    """)

def plot_follower_graph(next_words):
    import pydot

    graph = pydot.Dot("follower_graph", ordering="in")
    def mangle(s):
        #if isinstance(s, tuple) and len(s) == 1:
        #    s = s[0]
        return repr(s).replace(r'\n', r'\\n')
    for context, followers in next_words.items():
        graph.add_node(pydot.Node(mangle(context)))
        for follower in followers:
            edge = graph.add_edge(pydot.Edge(mangle(context), mangle(follower)))
            # A bit of a hack
            #if hasattr(followers, 'get'):
            #    edge.set_label(followers.get(follower))
            #else:
            #    count = None
            
    svg = graph.create_svg().decode('utf-8')
    return graph_out(svg)

def plot_follower_context_graph(next_words):
    # TODO: This is fugly. Use dot
    import pydot

    graph = pydot.Dot("follower_graph", ordering="in", strict=True)
    def mangle(s):
        #if isinstance(s, tuple) and len(s) == 1:
        #    s = s[0]
        return repr(s).replace(r'\n', r'\\n')
    for context, followers in next_words.items():
        #graph.add_node(pydot.Node(mangle(context)))
        for follower in followers:
            # A bit of a hack
            #edge = graph.add_edge(pydot.Edge(mangle(context), mangle(follower)))
            new_context = (*context[1:], follower)
            for follower in next_words.get(context, []):
                follower_context = (*context[1:], follower)
                graph.add_edge(pydot.Edge(
                    mangle(context),
                    mangle(follower_context),
                    label=mangle(follower)
                ))

    svg = graph.create_svg().decode('utf-8')
    return graph_out(svg)

def generate_tokens(next_words, context=None, max_tokens=200, seed=3):
    rng = random.Random(seed)

    if context is None:
        context = next(iter(next_words.keys()))
    yield from context

    for i in range(max_tokens):
        candidates = next_words.get(context, None)
        if not candidates: return

        choices, counts = zip(*candidates.items())
        if not choices: return
        next_word = rng.choice(choices)
        if next_word == '\n\n': return
        yield next_word
        context = (*context[1:], next_word)

# Doing this more succintly now
def get_ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        yield tokens[i:i+n]

def get_next_token_table(tokens, context_length, table=None):
    if table is None:
        table = defaultdict(Counter)
    for *context, next_token in get_ngrams(tokens, context_length + 1):
        table[tuple(context)][next_token] += 1
    
    return table

happy_birthday_text = """
Happy birthday to you 
Happy birthday to you 
Happy birthday dear Dave 
Happy birthday to you 
"""

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

blowin_text_finnish = """
Niin, ja kuinka monta tietä miehen täytyy kävellä, ennen kuin kutsut häntä mieheksi?
Ja kuinka monta merta valkoisen kyyhkysen täytyy purjehtia, ennen kuin se nukkuu hiekkaan?
Kyllä, ja kuinka monta kertaa kanuunankuulat täytyy lentää, ennen kuin ne on ikuisesti kielletty?

Kyllä, ja kuinka monta vuotta vuoren on oltava olemassa, ennen kuin se huuhtoutuu mereen?
Ja kuinka monta vuotta jotkut ihmiset voivat olla olemassa ennen kuin he saavat olla vapaita?
Kyllä, ja kuinka monta kertaa ihminen voi kääntää päätään ja teeskennellä, ettei hän vain näe?

Kyllä, ja kuinka monta kertaa miehen täytyy katsoa ylös, ennen kuin hän voi nähdä taivaan?
Ja kuinka monta korvaa yhdellä ihmisellä pitää olla, ennen kuin hän voi kuulla ihmisten itkevän?
Kyllä, ja kuinka monta kuolemaa kestää, ennen kuin hän tietää, että liian monta ihmistä on kuollut?
"""

blowin_text_german = """
Ja, und wie viele Wege muss ein Mann gehen, bevor man ihn einen Mann nennt?
Und wie viele Meere muss eine weiße Taube durchsegeln, bevor sie im Sand schläft?
Ja, und wie oft müssen die Kanonenkugeln fliegen, bevor sie für immer verboten werden?

Ja, und wie viele Jahre muss ein Berg existieren, bevor er ins Meer gespült wird?
Und wie viele Jahre können manche Menschen existieren, bevor sie frei sein dürfen?
Ja, und wie oft kann ein Mann den Kopf drehen und so tun, als würde er einfach nichts sehen?

Ja, und wie oft muss ein Mensch nach oben schauen, bevor er den Himmel sehen kann?
Und wie viele Ohren muss ein Mann haben, bevor er Menschen weinen hören kann?
Ja, und wie viele Todesfälle wird es dauern, bis er weiß, dass zu viele Menschen gestorben sind?
"""

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