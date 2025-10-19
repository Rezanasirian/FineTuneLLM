import random

class DataAugmentor:
    def augment_persian_query(self, query: str) -> str:
        synonyms = {
            "چقدر": ["چه مقدار", "میزان"],
            "است": ["می‌باشد", "هست"],
        }
        words = query.split()
        augmented = [synonyms.get(w, [w])[0] if random.random() < 0.3 else w for w in words]
        return " ".join(augmented)