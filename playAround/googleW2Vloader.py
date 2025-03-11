import gensim.downloader as api

def get_google_W2V_model():
    return api.load("word2vec-google-news-300")
