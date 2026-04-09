
# using newmodels in app.py (not giving good results in the similar products)
# models are giving perfect results
from flask import Flask, request, render_template, session, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# =======================================================================================
# 1. LOAD DATA & CLEAN
# =======================================================================================
train_data = pd.read_csv("newmodels/clean_data.csv")
train_data.drop_duplicates(subset=['Name'], inplace=True)

# Ensure numeric data for calculations
train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce').fillna(0)
train_data['ReviewCount'] = pd.to_numeric(train_data['ReviewCount'], errors='coerce').fillna(0)
train_data.reset_index(drop=True, inplace=True)

# =======================================================================================
# 2. COLLABORATIVE FILTERING (Bayesian Average Popularity)
# =======================================================================================
C = train_data['Rating'].mean() # Mean rating across the whole database
m = train_data['ReviewCount'].quantile(0.5) # Minimum reviews required to be considered "popular"

def bayesian_rating(x):
    v = x['ReviewCount']
    R = x['Rating']
    # Bayesian formula prevents 5-star items with 1 review from beating 4.8-star items with 1000 reviews
    if (v + m) > 0:
        return (v / (v + m) * R) + (m / (m + v) * C)
    return 0

train_data['Collab_Score'] = train_data.apply(bayesian_rating, axis=1)

# Normalize the Collaborative Score between 0 and 1
max_collab = train_data['Collab_Score'].max()
min_collab = train_data['Collab_Score'].min()
if max_collab > min_collab:
    train_data['Normalized_Collab'] = (train_data['Collab_Score'] - min_collab) / (max_collab - min_collab)
else:
    train_data['Normalized_Collab'] = 0

# GLOBAL TRENDING: Strictly sorted top-to-bottom by real Bayesian Popularity
global_top_products = train_data.sort_values(by='Collab_Score', ascending=False)

# =======================================================================================
# 3. CONTENT-BASED FILTERING (NLP TF-IDF)
# =======================================================================================
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    return text

# --- ENGINE 1: Strict Search Engine ---
def search_products(query, top_k=10):
    search_query = str(query).lower()
    matches = train_data[
        train_data['Name'].str.lower().str.contains(search_query, na=False) |
        train_data['Tags'].str.lower().str.contains(search_query, na=False)
    ].copy()

    if matches.empty:
        return pd.DataFrame()

    # STRICT SORT: Highest True Rating/Popularity wins
    matches = matches.sort_values(by='Collab_Score', ascending=False)
    return matches.head(top_k)

# --- ENGINE 2: HYBRID RECOMMENDER SYSTEM (For Product Page) ---
def get_recommendations(item_name, top_n=4):
    search_query = str(item_name).lower()
    
    # Find exact item first
    matches = train_data[train_data['Name'].str.lower() == search_query]
    if matches.empty:
        matches = train_data[train_data['Name'].str.lower().str.contains(search_query, na=False)]
        
    if matches.empty:
        return pd.DataFrame()

    item_index = matches.index[0]
    
    # Step 1: Get Content Similarity Math
    target_vector = tfidf_matrix_content[item_index]
    similarities = cosine_similarity(target_vector, tfidf_matrix_content).flatten()
    
    temp_df = train_data.copy()
    temp_df['Similarity_Score'] = similarities
    
    # Step 2: The Hybrid Formula (70% Similarity + 30% Popularity)
    temp_df['Hybrid_Score'] = (temp_df['Similarity_Score'] * 0.7) + (temp_df['Normalized_Collab'] * 0.3)
    
    # Filter out the searched item itself and absolutely irrelevant items
    temp_df = temp_df[(temp_df.index != item_index) & (temp_df['Similarity_Score'] > 0.05)]
    
    if temp_df.empty:
        return pd.DataFrame()
        
    # STRICT SORT: The highest Hybrid Score guarantees a highly similar AND highly rated item
    recommended_items = temp_df.sort_values(by='Hybrid_Score', ascending=False)
    
    return recommended_items.head(top_n)

# =======================================================================================
# 4. ROUTES
# =======================================================================================

@app.route("/")
@app.route("/index")
def index():
    if 'history' not in session or not session['history']:
        products_to_show = global_top_products.head(8)
    else:
        most_recent_item = session['history'][-1]
        personalized_recs = get_recommendations(most_recent_item, top_n=4)
        
        if personalized_recs.empty:
            products_to_show = global_top_products.head(8)
        else:
            needed_fillers = 8 - len(personalized_recs)
            fillers = global_top_products[~global_top_products['Name'].isin(personalized_recs['Name'])].head(needed_fillers)
            products_to_show = pd.concat([personalized_recs, fillers])
            
    return render_template('index.html', trending_products=products_to_show, truncate=truncate)

@app.route("/product")
def product():
    name = request.args.get('name')
    if not name:
        return redirect(url_for('index'))
        
    if 'history' not in session:
        session['history'] = []
    if not session['history'] or session['history'][-1] != name:
        session['history'].append(name)
        if len(session['history']) > 10:
            session['history'].pop(0)
        session.modified = True

    product_data = train_data[train_data['Name'] == name]
    if product_data.empty:
        return redirect(url_for('index'))
        
    product_details = product_data.iloc[0]
    similar_items = get_recommendations(name, top_n=4)
    
    return render_template('product.html', product=product_details, similar_items=similar_items, truncate=truncate)

@app.route("/add_to_cart", methods=['POST'])
def add_to_cart():
    name = request.form.get('name')
    if 'cart' not in session:
        session['cart'] = []
    if name and name not in session['cart']:
        session['cart'].append(name)
        session.modified = True
    return redirect(url_for('cart'))

@app.route("/remove_from_cart", methods=['POST'])
def remove_from_cart():
    name = request.form.get('name')
    if 'cart' in session and name in session['cart']:
        session['cart'].remove(name)
        session.modified = True
    return redirect(url_for('cart'))

@app.route("/cart")
def cart():
    if 'cart' not in session:
        session['cart'] = []
    cart_items = train_data[train_data['Name'].isin(session['cart'])]
    return render_template('cart.html', cart_items=cart_items, truncate=truncate)

@app.route("/history")
def history():
    if 'history' not in session:
        session['history'] = []
    history_items = []
    seen = set()
    for name in reversed(session['history']):
        if name not in seen:
            item_data = train_data[train_data['Name'] == name]
            if not item_data.empty:
                history_items.append(item_data.iloc[0])
            seen.add(name)
    return render_template('history.html', history_items=history_items, truncate=truncate)

@app.route("/search", methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        
        # New Logic: Grab the typed input. If it's empty or invalid, default to 10.
        top_k_input = request.form.get('top_k', '').strip()
        top_k = int(top_k_input) if top_k_input.isdigit() and int(top_k_input) > 0 else 10
        
        search_results = search_products(query, top_k=top_k)

        if search_results.empty:
            message = f"No results found for '{query}'."
            return render_template('main.html', message=message)
        else:
            return render_template('main.html', search_results=search_results, query=query, truncate=truncate)
            
    return render_template('main.html')

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')
