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
        color = colors[i%len(colors)]
        # TODO: Be more general!
        if string == ' ':
            decoration = "underline"
        else:
            decoration = "none"

        n_newlines = string.count('\n')
        string = string.replace("\n", "\\n")
        string += "\n"*n_newlines

        out += f'<span style="color: {color}; text-decoration: {decoration}">{string}</span>'
    
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

def generate_tokens(next_words, context=None, max_tokens=50, seed=3):
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

