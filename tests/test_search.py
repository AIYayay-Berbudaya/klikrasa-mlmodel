from src.models.suggestion import SemanticSearch

def test_basic_search_runs():
    e = SemanticSearch()
    results = e.search("kue manis", top_k=3)
    assert isinstance(results, list)

def test_search_similar_to():
    e = SemanticSearch()
    # if dataset present, try to call similar for first item's id
    if len(e.dataset) > 0:
        first_id = e.dataset[0].get("id")
        res = e.search_similar_to(first_id, top_k=2)
        assert isinstance(res, list)