from tokrag.collect.acl_anthology import _parse_entry, _strip_braces


def test_strip_braces_keeps_text_removes_protection_braces():
    assert _strip_braces("{S}upra{T}ok: Cross-Boundary Tokenization") == "SupraTok: Cross-Boundary Tokenization"


def test_parse_entry_extracts_core_fields():
    body = (
        'title = "{S}upra{T}ok: Cross-Boundary Tokenization",\n'
        'author = "Doe, Jane  and\n'
        "      Smith, John\",\n"
        'booktitle = "Proceedings of ACL",\n'
        'year = "2026",\n'
        'url = "https://aclanthology.org/2026.tacl-1.80/",\n'
        'abstract = "We study tokenization boundaries."\n'
    )
    paper = _parse_entry("doe-2026-supratok", body)
    assert paper is not None
    assert paper.title == "SupraTok: Cross-Boundary Tokenization"
    assert paper.authors == ["Doe, Jane", "Smith, John"]
    assert paper.year == "2026"
    assert paper.venue == "Proceedings of ACL"
    assert "tokenization boundaries" in paper.abstract


def test_parse_entry_with_no_title_returns_none():
    assert _parse_entry("k", 'year = "2020"') is None
