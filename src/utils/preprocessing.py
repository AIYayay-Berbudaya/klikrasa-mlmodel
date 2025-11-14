import orjson
from typing import Dict, List

def read_jsonl(path):
    items = []
    with open(path, 'rb') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(orjson.loads(line))
    return items


def normalize_text(s: str) -> str:
    s = s or ''
    return ' '.join(s.replace('\n', ' ').split())


def kue_to_corpus(kue: Dict) -> str:
    parts = [kue.get('title', ''), kue.get('description', ''), kue.get('history', ''), kue.get('making_process', '')]
    return normalize_text(' '.join(parts))