from transformers import BartTokenizer, BartForConditionalGeneration
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx

def preprocessing(text: str) -> str:
    text = text.replace('\n', ' ')

# abstractive summary generation
def ab_summary(text: str) -> str:
    print('working')
    checkpoint = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    model = BartForConditionalGeneration.from_pretrained(checkpoint)
    input_ids = tokenizer.batch_encode_plus(
        [text],
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    )
    summary_ids = model.generate(
        input_ids['input_ids']
    )
    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

# extractive summary generation
def read_article(text: str) -> list:
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence.replace("[^a-zA-Z0-9]", " ")
    return sentences

def sentence_similarity(sent1: str, sent2: str, stopwords=None) -> int:
    # creating vectors from the sentences and calculating cosine similarity between these vectors

    if stopwords is None:
        stopwords = []
    sent1 = [word.lower() for word in sent1]
    sent2 = [word.lower() for word in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    # build vector for first sentence
    for word in sent1:
        if word not in stopwords:
            vector1[all_words.index(word)] += 1
    # build vector for second sentence
    for word in sent2:
        if word not in stopwords:
            vector2[all_words.index(word)] += 1

    return 1-cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences: list, stop_words: list) -> list:
    # creating similarity matrix to store the similarity values
    
    #create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix 

def ex_summary(text: str, top_n=5) -> list:
    # main function putting everything together

    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = stopwords.words('english')

    # read text and tokenize
    sentences = read_article(text)

    # generate similarity matrix
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # sort rank and place top sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # get top n number of sentences based on rank
    summarise_text = [ranked_sentences[i][1] for i in range(top_n)]

    # ouput summarised version
    return " ".join(summarise_text)