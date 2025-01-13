"""
@authors: Taxiarchis Boumpas 21390151
          Thanasis Moutzouris 21390137    
"""

# LIBRARIES
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# WEB CRAWLER
def fetch_wikipedia_articles(query, limit):
    # COLLECTING WIKIPEDIA ARTICLES
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching articles for query '{query}'. HTTP Status Code: {response.status_code}")
        return []
    data = response.json()
    articles = data.get('query', {}).get('search', [])
    return [f"https://en.wikipedia.org/wiki/{article['title'].replace(' ', '_')}" for article in articles]

def web_crawler(query, limit):
    urls = fetch_wikipedia_articles(query, limit)
    if not urls:
        print(f"No articles found for query '{query}'.")
        return [], {}

    academic_papers = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # TITLE
            title = soup.find('h1', id='firstHeading').text.strip()

            # CONTENT
            content_div = soup.find('div', id='mw-content-text')
            paragraphs = content_div.find_all('p') if content_div else []
            if paragraphs:
                content = ' '.join([p.text for p in paragraphs[:3]])  # Λήψη των πρώτων 3 παραγράφων
            else:
                content = "No content available"

            # DATA STRUCTURE
            paper = {
                'title': title,
                'content': text_processing(content)
            }
            academic_papers.append(paper)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    # JSON FILE
    with open(f'{query}_wikipedia_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(academic_papers, json_file, ensure_ascii=False, indent=2)

    print(f"Collected {len(academic_papers)} articles.")
    return academic_papers, create_inverted_index(academic_papers)

# TEXT PROCESSING
def text_processing(text):
    # TOKENIZATION
    tokens = word_tokenize(text)
    
    # ELIMINATING STOPWORDS
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # LEMMATIZATION
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# INVERTED INDEX
def create_inverted_index(academic_papers):
    inverted_index = {}
    for idx, paper in enumerate(academic_papers):
        for field, values in paper.items():
            if isinstance(values, list):
                for term in values:
                    term_str = str(term)  #Converting term to a string
                    if term_str not in inverted_index:
                        inverted_index[term_str] = set()
                    inverted_index[term_str].add(idx)
    return inverted_index

# SEARCH ENGINE
def search_engine(academic_papers, inverted_index, query):
    algorithm = input("\nAvailable retrieval algorithms (BOOLEAN, VSM, BM25)\nChoose: ").upper()
    if algorithm not in ['BOOLEAN', 'VSM', 'BM25']:
        print("Error! Invalid algorithm choice. Please choose: Boolean, VSM or BM25\n")
        return
    
    matching_docs = retrieve_documents(query, algorithm, inverted_index, academic_papers)
    if matching_docs is not None:
        display_results(matching_docs)
        relevant_docs = get_relevant_docs(academic_papers)
        evaluate_system(matching_docs, relevant_docs, len(academic_papers))
    else:
        print("Error! An issue occurred during retrieval.\n")

def retrieve_documents(query, algorithm, inverted_index, academic_papers):
    if algorithm == 'BOOLEAN':
        return boolean_retrieval(query, inverted_index, academic_papers)
    elif algorithm == 'VSM':
        return vsm_retrieval(query, academic_papers)
    elif algorithm == 'BM25':
        return bm25_retrieval(query, academic_papers, k1=1.5, b=0.75)
    else:
        return []

def display_results(matching_docs):
    # SHOW RESULTS 
    if not matching_docs:
        print("No matching documents found.\n")
    else:
        print("Matching documents:\n")
        for i, doc in enumerate(matching_docs, start=1):
            print(f"{i}. Title: {doc['title']}")
            content_preview = ' '.join(doc['content'])[:900] if doc['content'] else "No content available"
            print(f"   Content: {content_preview}\n")  # Εμφάνιση πρώτων 900 χαρακτήρων

def get_relevant_docs(all_docs):
    # SELECT RELEVANT DOCUMENTS
    print("\nSelect the relevant document numbers, separated by commas (e.g., 1,3,5):")
    user_input = input("Relevant documents: ").strip()
    relevant_indices = [int(num) - 1 for num in user_input.split(",") if num.isdigit() and 0 < int(num) <= len(all_docs)]
    return set(relevant_indices)

def evaluate_system(matching_docs, relevant_docs, total_docs):
    # PERFMORMANCE OF THE SYSTEM
    retrieved_indices = set(range(len(matching_docs)))

    true_positives = len(relevant_docs & retrieved_indices)
    false_positives = len(retrieved_indices - relevant_docs)
    false_negatives = len(relevant_docs - retrieved_indices)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # MAP CALCULATION 
    average_precision = sum([(i + 1) / (rank + 1) for rank, i in enumerate(sorted(relevant_docs & retrieved_indices))]) / len(relevant_docs) if relevant_docs else 0

    print("\nSystem Performance Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"MAP (Mean Average Precision): {average_precision:.2f}")

def boolean_retrieval(query, inverted_index, docs):
    query_terms = query.split()

    # EMPTYING SET FOR THE FINAL RESULT
    matching_docs = set(range(len(docs)))

    # SETTING UP A MARKER TO CHECK IF THE NEXT WORD NEEDS TO BE EXCLUDED.
    negate_next = False

    # SETTING THE LOGICAL OPERATOR TO NONE
    operator = None

    for term in query_terms:
        if term == 'NOT':
            negate_next = True
            continue

        # CHECKING FOR POSSIBLE LOGICAL OPERATOR
        if term in ['AND', 'OR']:
            operator = term
            continue  
        # TERM TO A STRING
        term_str = str(term)  

        # BOOLEAN OPERATIONS (NOT, AND, OR)
        if term_str in inverted_index:
            term_results = set(inverted_index[term_str])
            if negate_next:
                matching_docs -= term_results
                negate_next = False
            else:
                # EXECUTE THE SPECIFIED LOGICAL OPERATION.
                if operator == 'AND':
                    matching_docs = matching_docs.intersection(term_results)
                elif operator == 'OR':
                    matching_docs = matching_docs.union(term_results)
                else:
                    matching_docs = term_results

    # RETURN THE FINAL DOCUMENT SET
    return [docs[doc_id] for doc_id in matching_docs]

def vsm_retrieval(query, docs):
    if not docs:
        print("Error! No documents available for retrieval.\n")
        return []
    
    # LIST OF DOCUMENT TITLES
    doc_titles = [' '.join(doc['title']) + ' '.join(doc.get('abstract', [])) for doc in docs]
    
    if not doc_titles:
        print("Error! No document titles available for retrieval.\n")
        return []
    
    vectorizer = TfidfVectorizer()
    term_doc_matrix = vectorizer.fit_transform([query] + doc_titles)
    
    if term_doc_matrix.shape[1] < 2:
        print("Error! Not enough documents for retrieval.\n")
        return []
    
    # COSINE SIMILARITIES
    cos_similarities = cosine_similarity(term_doc_matrix[0:1], term_doc_matrix[1:]).flatten()
    
    # ORGANIZING DOCUMENT IDS (BASED ON SIMILARITY)
    ranked_doc_ids = np.argsort(cos_similarities)[::-1]
    
    # DISPLAY MATCHING DOCUMENTS (EXCLUDE THE QUERY DOCUMENT)
    matching_docs = [docs[doc_id] for doc_id in ranked_doc_ids[1:] if cos_similarities[doc_id] > 0]
    
    return matching_docs

def bm25_retrieval(query, docs, k1, b):
    if not docs:
        print("Error! No documents available for retrieval.\n")
        return []
    
    # LIST OF DOCUMENT TITLES
    doc_titles = [' '.join(doc['title']) + ' '.join(doc.get('abstract', [])) for doc in docs]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + doc_titles)
    doc_length = np.sum(tfidf_matrix > 0, axis=1)
    avg_doc_length = np.mean(doc_length)
    idf = vectorizer.idf_
    
    k_broadcasted = k1 * ((1 - b) + b *(doc_length / avg_doc_length))
        
    bm25_scores = (tfidf_matrix[0] * (k1 + 1)) / (tfidf_matrix[0] + k_broadcasted[0, :]) * idf
    
    # ORGANIZING DOCUMENT IDS (BASED ON SIMILARITY)
    ranked_doc_ids = np.argsort(bm25_scores)[::-1]
    
    return [docs[doc_id] for doc_id in ranked_doc_ids]

# EXECUTION
query = input("Enter your query for Wikipedia search: ")  # Εισαγωγή query από τον χρήστη
limit = int(input("Enter the ammount of articles you want to search for: "))  # Εισαγωγή επιθυμητού αριθμού άρθρων
academic_papers, inverted_index = web_crawler(query, limit)
search_engine(academic_papers, inverted_index, query)
