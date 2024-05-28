import marimo as mo

from pprint import pformat
from collections import defaultdict, Counter

_pre_box_height = "10em";
def pre_box(text):
    return mo.Html(f"""
<pre class="pre_out_box">
{text}
</pre>""")

def python_out(code):
    return mo.Html(f"""
<pre class="python_out_box">
{pformat(code, sort_dicts=False, compact=True)}
</pre>""")

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

def graph_out(svg):
    return mo.Html(f"""
        <div style="overflow: auto; max-height: 32em;">
        {svg}
        </div>
    """)

def plot_follower_graph(next_words):
    # TODO: This is fugly. Use dot
    import pydot

    graph = pydot.Dot("follower_graph", ordering="in")
    def mangle(s):
        #if isinstance(s, tuple) and len(s) == 1:
        #    s = s[0]
        return repr(s).replace(r'\n', r'\\n')
    for context, followers in next_words.items():
        # TODO: Fix for 
        graph.add_node(pydot.Node(mangle(context)))
        for follower in followers:
            graph.add_edge(pydot.Edge(mangle(context), mangle(follower)))
    svg = graph.create_svg().decode('utf-8')

    return graph_out(svg)

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
