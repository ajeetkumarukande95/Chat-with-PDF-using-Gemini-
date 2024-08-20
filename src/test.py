import pytest

def test_get_text_chunks():
    text = "Sample text"
    chunks = get_text_chunks(text)
    assert len(chunks) > 0
