import spacy

# Load medium English model
nlp = spacy.load("en_core_web_md")

# Description of Planet Hulk
planet_hulk = """Will he save their world or destroy it? When the Hulk becomes too
dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into
space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on
the planet Sakaar where he is sold into slavery and trained as a gladiator."""

# Convert the description into a spaCy document
planet_doc = nlp(planet_hulk)


# Read movie descriptions from file
with open("movies.txt", "r") as file:
    movies = file.readlines()


# Function to find the most similar movie
def next_movie(description):

    description_doc = nlp(description)

    highest_similarity = 0
    best_movie = ""

    # Compare Planet Hulk description to every movie
    for movie in movies:

        title, movie_description = movie.split(":", 1)

        movie_doc = nlp(movie_description)

        similarity = description_doc.similarity(movie_doc)

        # Track the highest similarity score
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_movie = title

    return best_movie


# Print recommendation
print("Next movie to watch:", next_movie(planet_hulk))

'''Movie Recommendation System This program recommends a movie based on the similarity of its description
to the movie "Planet Hulk".

The spaCy language model converts movie descriptions into vector
representations. These vectors allow us to calculate semantic similarity
between texts.

The program compares the description of Planet Hulk with each movie
description in the dataset and returns the title of the movie with the
highest similarity score.

This demonstrates how natural language processing can be used in
recommendation systems similar to those used by platforms like Netflix.'''