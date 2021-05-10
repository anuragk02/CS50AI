import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    documents = {}
    
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        with open(path, "r", encoding="utf8") as f:
            documents[file] = f.read()
            
    return documents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.lower()
    words = nltk.word_tokenize(document)
    
    for word in words:
        if word in nltk.corpus.stopwords.words("english"):
            words.remove(word)
        else:
            for char in word:
                if char in string.punctuation:
                    word.replace(char,'')
                    
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    doc_counts = {}
    idf_vals = {}
    
    for document in documents:
        for word in set(documents[document]):
            try:
                doc_counts[word] += 1
            except KeyError:
                doc_counts[word] = 1
    
    for word in doc_counts:
        idf_vals[word] = math.log(len(documents)/doc_counts[word])
    
    return idf_vals


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidf_vals = {}
    
    for file in files:
        for word in query:
            try:
                tfidf_vals[file] += files[file].count(word) * idfs[word]
            except KeyError:
                tfidf_vals[file] = files[file].count(word) * idfs[word]
                
    ranked_list = sorted(tfidf_vals, key=tfidf_vals.get,reverse=True)
    
    return ranked_list[0:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ranked_list = list()

    for sentence in sentences:
        sentence_variables = [sentence, 0, 0]

        for word in query:
            if word in sentences[sentence]:
                sentence_variables[1] += idfs[word]
                sentence_variables[2] += sentences[sentence].count(word) / len(sentences[sentence])

        ranked_list.append(sentence_variables)
        
    return [sentence for sentence, word_count, q_term_density in sorted(ranked_list, key=lambda variable: (variable[1], variable[2]), reverse=True)][:n]
    

if __name__ == "__main__":
    main()
