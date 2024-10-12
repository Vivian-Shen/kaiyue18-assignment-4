from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
# Fetch dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize vectorizer (TF-IDF)
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000)
doc_term_matrix = vectorizer.fit_transform(documents)

# Perform SVD (LSA)
svd = TruncatedSVD(n_components=100)
lsa_matrix = svd.fit_transform(doc_term_matrix)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 
    # Transform the query using the same TF-IDF vectorizer
    # Transform the query using the same TF-IDF vectorizer
    query_tfidf = vectorizer.transform([query])
    
    # Project the query into the LSA space
    query_lsa = svd.transform(query_tfidf)
    
    # Calculate cosine similarity between the query and all documents in LSA space
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]
    
    # Get the indices of the top 5 most similar documents
    top_indices = np.argsort(similarities)[::-1][:5]
    
    # Retrieve the top 5 similar documents and their similarity scores
    top_documents = [documents[i] for i in top_indices]
    top_similarities = similarities[top_indices]
    
    # Return results as Python lists to ensure JSON serializability
    return top_documents, top_similarities.tolist(), top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
