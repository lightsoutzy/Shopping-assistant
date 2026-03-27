from app.services.intent_router import classify


def test_greeting_is_general_chat():
    assert classify("hello", False) == "general_chat"
    assert classify("hi there", False) == "general_chat"


def test_capability_question_is_general_chat():
    assert classify("what can you do", False) == "general_chat"
    assert classify("who are you", False) == "general_chat"


def test_product_keyword_is_recommend():
    assert classify("find me a black jacket", False) == "recommend_products"
    assert classify("show me sneakers under 80", False) == "recommend_products"
    assert classify("recommend a t-shirt for sports", False) == "recommend_products"


def test_color_keyword_is_recommend():
    assert classify("I want something blue", False) == "recommend_products"


def test_vague_short_input_is_clarify():
    assert classify("ok", False) == "clarify_request"
    assert classify("hi", False) in ("general_chat", "clarify_request")


def test_image_always_routes_to_image_search():
    assert classify("find something like this", True) == "image_search"
    assert classify("", True) == "image_search"
    assert classify("but make it casual", True) == "image_search"
