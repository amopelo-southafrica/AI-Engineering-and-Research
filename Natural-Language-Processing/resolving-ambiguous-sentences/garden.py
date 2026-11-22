import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# List of garden path sentences
gardenpathSentences = [
    "The old man the boat.",
    "The horse raced past the barn fell.",
    "Mary gave the child a Band-Aid.",
    "That Jill is never here hurts.",
    "The cotton clothing is made of grows in Mississippi."
]

# Process each sentence
for sentence in gardenpathSentences:

    print("\nSentence:", sentence)

    doc = nlp(sentence)

    # Tokenisation
    print("Tokens:")
    for token in doc:
        print(token.text)

    # Named Entity Recognition
    print("Entities:")
    for ent in doc.ents:
        print(ent.text, "|", ent.label_, "|", spacy.explain(ent.label_))


# Examples of entity explanations
print("\nEntity explanations:")
print("GPE:", spacy.explain("GPE"))
print("PERSON:", spacy.explain("PERSON"))
print("ORG:", spacy.explain("ORG"))