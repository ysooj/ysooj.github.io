from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from scipy.spatial.distance import cosine

sentences = [
    "The quick brown fox jumps over the lazy dog",
    "I love playing whi my pet dog",
    "The dog barks at the stranger",
    "The cat sleeps on the sofa"
]

processed = [simple_preprocess(sentence) for sentence in sentences]
print(processed)

model = Word2Vec(sentences=processed, vector_size=5, window=5, min_count=1, sg=0)
dog = model.wv['dog']
cat = model.wv['cat']

sim = 1 - cosine(dog, cat)
print(sim)