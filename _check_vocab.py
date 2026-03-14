from pipeline.utils import _default_class_names
cn = _default_class_names()
print(f"total={len(cn)}, unique={len(set(cn))}")
idx = {n: i for i, n in enumerate(cn)}
for term in ["edge", "edge highlight", "edge blur", "gray", "thirst"]:
    print(f"  {term!r} -> {idx.get(term, 'MISSING')}")
