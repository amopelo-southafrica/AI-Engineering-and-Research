import spacy

# Load medium English language model
nlp = spacy.load("en_core_web_md")

# ---------------- WORD SIMILARITY ---------------- #

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print("Word Similarity Examples:\n")

print("cat vs monkey:", word1.similarity(word2))
print("banana vs monkey:", word3.similarity(word2))
print("banana vs cat:", word3.similarity(word1))


# ---------------- TOKEN COMPARISON ---------------- #

print("\nToken Similarity Comparisons:\n")

tokens = nlp("cat apple monkey banana")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# ---------------- SENTENCE SIMILARITY ---------------- #

sentence_to_compare = "Why is my cat on the car"

sentences = [
    "where did my dog go",
    "Hello, there is my car",
    "I've lost my car in my car",
    "I'd like my boat back",
    "I will name my dog Diana"
]

model_sentence = nlp(sentence_to_compare)

print("\nSentence Similarity Results:\n")
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence, "-", similarity)

#The similarity between "cat" and "monkey" is higher than between "cat" and "banana".
# This makes sense because cat and monkey are both animals, while banana is a fruit.Similarly, "banana" and "monkey" have a moderate similarity because monkeys arecommonly associated with bananas.
# This demonstrates that spaCy's language model understands semantic relationshipsbetween words based on contextual meaning.