from classbot.llm.web_supplement import parse_web_supplement


def test_parse_web_supplement_accepts_valid_claims():
    raw = '{"claims":[{"text":"一般背景。","web_ids":["W1"]}]}'
    claims = parse_web_supplement(raw, {"W1", "W2"})
    assert len(claims) == 1
    assert claims[0].text == "一般背景。"
    assert claims[0].web_ids == ("W1",)


def test_parse_web_supplement_rejects_unknown_ids():
    raw = '{"claims":[{"text":"一般背景。","web_ids":["W9"]}]}'
    assert parse_web_supplement(raw, {"W1"}) == ()


def test_parse_web_supplement_rejects_inline_tags():
    raw = '{"claims":[{"text":"一般背景[S1]。","web_ids":["W1"]}]}'
    assert parse_web_supplement(raw, {"W1"}) == ()


def test_parse_web_supplement_rejects_fullwidth_and_lowercase_tags():
    fullwidth = '{"claims":[{"text":"一般背景［W1］。","web_ids":["W1"]}]}'
    lowercase = '{"claims":[{"text":"一般背景[w1]。","web_ids":["W1"]}]}'
    assert parse_web_supplement(fullwidth, {"W1"}) == ()
    assert parse_web_supplement(lowercase, {"W1"}) == ()


def test_parse_web_supplement_rejects_bad_shape_and_empty_claims():
    assert parse_web_supplement('{"claims":[]}', {"W1"}) == ()
    assert parse_web_supplement('{"claims":[{"text":"x"}]}', {"W1"}) == ()
    assert parse_web_supplement("not json", {"W1"}) == ()
