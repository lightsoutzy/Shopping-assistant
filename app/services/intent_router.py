"""
Rule-based intent classifier. No LLM used here.

Intent labels (legacy / single-turn):
  general_chat        - greetings, identity, capability questions
  recommend_products  - text-based product search / filtering
  image_search        - any request that includes an image
  clarify_request     - extremely vague text with no image

Additional intents for conversational / context-aware routing:
  new_text_search             - fresh product search (replaces active context)
  followup_on_current_results - question/filter on currently shown products
  out_of_scope                - unrelated question; should be refused
  reset_context               - user wants to clear and start over
"""

import re

# Patterns that strongly signal a product recommendation request
_RECOMMEND_RE = re.compile(
    r"\b("
    r"recommend|suggest|find|show|look for|looking for|search|need|want|get me"
    r"|t-shirt|tshirt|shirt|shoes|sneaker|jacket|hoodie|sweatshirt|shorts|bag|handbag|backpack"
    r"|black|white|blue|red|green|grey|gray|brown|pink|yellow|navy|beige"
    r"|men.s|women.s|male|female|unisex"
    r"|casual|sports|formal|summer|winter|spring|fall"
    r"|under \$|less than|below|cheap|affordable|budget"
    r")\b",
    re.IGNORECASE,
)

# Patterns that signal general conversation
_CHAT_RE = re.compile(
    r"^(hi|hello|hey|howdy)\b"
    r"|\b(what (is|are|can) you|who are you|your name|what do you do|how (do|can) (i|you)|help me$)\b"
    r"|\b(what (kind|type|sort) of|what products|can you help|what can you)\b",
    re.IGNORECASE,
)

# Patterns that signal a follow-up on currently displayed products
_FOLLOWUP_RE = re.compile(
    r"\b("
    r"cheapest|least expensive|most expensive|priciest|lowest price|highest price"
    r"|which (is|are|one|would|of)"
    r"|of (these|those|them)"
    r"|only show|just show|only the|just the"
    r"|most (casual|formal|sporty|versatile|comfortable|affordable|expensive)"
    r"|more (casual|formal|sporty|versatile|comfortable|affordable)"
    r"|best (for|looking|one|option)"
    r"|looks? (best|better|most)"
    r"|compare|versus|vs\b"
    r"|between (these|those|them|the)"
    r"|filter (these|those|them|down)"
    r"|narrow (down|it)"
    r"|the \w+ one"
    r"|\bones?\b"   # "white ones?", "which one?", "the blue one"
    r")\b",
    re.IGNORECASE,
)

# Patterns that signal the user wants a new/different search, overriding
# follow-up detection even when active products exist.
_NEW_SEARCH_OVERRIDE_RE = re.compile(
    r"\b("
    r"actually|instead|something (else|different|new)|how about|what about"
    r"|different (batch|results|options|ones|items|selection)"
    r"|more (results|options|items)|next (batch|page|results|set)"
    r"|show me (more|other|different|another)"
    r"|other options|other items|other results"
    r"|search (the )?(full |whole )?catalog"
    r")\b",
    re.IGNORECASE,
)

# Patterns that signal user wants to reset / start over
_RESET_RE = re.compile(
    r"^(start over|reset|clear|start again|new search|begin again|never ?mind|forget it)\.?!?$",
    re.IGNORECASE,
)

# Patterns for clearly out-of-scope requests.
# No closing \b on the outer group — many patterns end with spaces or mid-word
# tokens that are incompatible with \b. The opening \b on each alternation is
# sufficient to avoid mid-word false positives.
_OUT_OF_SCOPE_RE = re.compile(
    r"("
    # Programming / CS / algorithms
    r"\blinked list\b|\bbinary (search|tree)\b|\bhash(map| map| table)\b"
    r"|\bheap sort\b|\bmerge sort\b|\bquick sort\b"
    r"|\btime complexity\b|\bbig.?o notation\b|\brecursion\b|\balgorithm\b"
    r"|\bhow (do|to) (reverse|sort|implement|debug|compile|traverse) "
    r"|\bstack overflow\b|\bgit (commit|push|pull|clone|rebase)\b"
    r"|\bdocker(file)?\b|\bkubernetes\b|\bsql (query|join|table|index)\b"
    r"|\bpython (syntax|function|class|loop)\b|\bjavascript\b|\btypescript\b"
    r"|\bjava\b|\bc\+\+\b|\brust\b|\bgolang\b"
    r"|\bhtml (tag|element)\b|\bcss (class|selector|property)\b"
    r"|\bwrite (a |me a |some )?(function|class|script|code|program|loop|query)\b"
    r"|\bdebug (this|my|the) (code|error|bug|function)\b"
    # Math / science
    r"|\bsolve (for|this equation|the equation)\b|\bintegral of\b|\bderivative of\b"
    r"|\blinear algebra\b|\bcalculus\b|\bdifferential equation\b"
    r"|\bquantum (physics|mechanics|computing)\b|\btheory of relativity\b"
    r"|\bperiodic table\b|\bchemical (formula|reaction|bond)\b|\bmolecular\b"
    r"|\bspeed of light\b|\bthermodynamics\b"
    # Geography / trivia / general knowledge
    r"|\bcapital (of|city of) "       # e.g. "capital of France"
    r"|\bwhat.s the weather\b|\bweather (in|for|today)\b"
    r"|\bwho (invented|discovered|created|wrote) "   # "who invented the ..."
    r"|\bhistory of "                 # "history of Rome"
    r"|\bwhen did "                   # "when did X happen"
    r"|\bwhat year (was|did|is)\b"
    r"|\bstock (price|market|exchange)\b|\bcryptocurrency\b|\bbitcoin\b|\bnft\b"
    r"|\blatest news\b|\bcurrent events\b|\bpolitics\b|\belection (results|winner)\b"
    r"|\bsports (score|result)\b|\bwho won the\b|\bworld cup\b|\bchampionship\b"
    r"|\bpresident of\b|\bprime minister of\b"
    # Non-shopping writing / life advice
    r"|\bwrite me a (poem|story|essay|song|novel|letter|speech)\b"
    r"|\bgive me a recipe\b|\bhow (do i |to )?cook "
    r"|\bmedical advice\b|\bsymptoms of\b|\bdiagno[sz]\b|\bprescription for\b"
    r"|\blife advice\b|\brelationship advice\b|\bshould i quit (my )?(job|school)\b"
    r"|\bmeaning of life\b|\bphilosophy of "
    r")",
    re.IGNORECASE,
)

_MIN_RECOMMEND_WORDS = 3  # queries shorter than this with no image → clarify


def classify(text: str, has_image: bool) -> str:
    """
    Legacy single-turn classifier. Preserved for backward compatibility and tests.
    Returns: general_chat | recommend_products | image_search | clarify_request
    """
    stripped = text.strip()

    if has_image:
        return "image_search"

    if _CHAT_RE.search(stripped):
        return "general_chat"

    if _RECOMMEND_RE.search(stripped):
        return "recommend_products"

    word_count = len(stripped.split())
    if word_count < _MIN_RECOMMEND_WORDS:
        return "clarify_request"

    return "recommend_products"


def classify_with_context(text: str, has_image: bool, active_products: list) -> str:
    """
    Context-aware classifier for multi-turn conversations.

    Returns one of:
      image_search | reset_context | out_of_scope | followup_on_current_results
      | general_chat | clarify_request | new_text_search
    """
    stripped = text.strip()

    if has_image:
        return "image_search"

    if _RESET_RE.match(stripped):
        return "reset_context"

    if _OUT_OF_SCOPE_RE.search(stripped):
        return "out_of_scope"

    # Follow-up only makes sense when there are active products in context
    # and the user isn't asking for a new search with different criteria
    if (
        active_products
        and _FOLLOWUP_RE.search(stripped)
        and not _NEW_SEARCH_OVERRIDE_RE.search(stripped)
    ):
        return "followup_on_current_results"

    if _CHAT_RE.search(stripped):
        return "general_chat"

    if _RECOMMEND_RE.search(stripped):
        return "new_text_search"

    word_count = len(stripped.split())
    if word_count < _MIN_RECOMMEND_WORDS:
        return "clarify_request"

    return "new_text_search"
