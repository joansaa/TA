from flask import Flask, request, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
import seaborn as sns
import base64
import os
from io import BytesIO
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/path/to/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_data(file_path):
    data = pd.read_excel(file_path)
    #column modification
    special_chars = [")", "(", "/", ".", " ","-"] # Karakter khusus yang akan diganti dan konversi nama kolom menjadi huruf kecil
    for char in special_chars:
        data.columns = data.columns.str.replace(char, "_")
    data.columns = data.columns.str.replace("__", "_") # menghapus kolom yang memiliki underscore dua diubah menjadi satu
    data.columns = data.columns.str.removesuffix("_") # menghapus underscore di akhir nama kolom
    data.columns = data.columns.str.lower()  # Mengonversi nama kolom menjadi huruf kecil
    # Daftar kolom yang ingin diubah menjadi huruf kecil
    columns_to_lower = [
        "no_pesanan",
        "nama_produk",
        "status_pesanan",
    ]
    
    #case folding
    # Mengubah isi kolom menjadi huruf kecil menggunakan str.lower()
    for column in columns_to_lower:
        data[column] = data[column].str.lower()

    #punctuation removal
    data["nama_produk"] = data["nama_produk"].str.translate(str.maketrans('', '', string.punctuation))
    data["nama_produk"] = data["nama_produk"].str.replace(r"[^\w\s]", "")
    data["nama_produk"] = data["nama_produk"].str.replace(r"\s+", " ")

    #normalization
    def normalize_text(data):
        words = data.split()
        normalized_words = []
        for i, word in enumerate(words):
            if i == 0:
                normalized_words.append(word.lower())
            else:
                normalized_words.append(word)
        return " ".join(normalized_words)

    data["nama_produk"] = data["nama_produk"].apply(normalize_text)

    def remove_duplicates(text):
        words = text.split()
        seen = set()
        removed_duplicates = []
        for word in words:
            if word not in seen:
                seen.add(word)
                removed_duplicates.append(word)
        return " ".join(removed_duplicates)

    data["nama_produk"] = data["nama_produk"].apply(remove_duplicates)

    #filtering
    # pesanan_selesai = data[data["status_pesanan"] == "Selesai"][["status_pesanan"]]
    data['pesanan_selesai'] = data[data["status_pesanan"] == "selesai"][["status_pesanan"]]

    #grouping
    data['tanggal_transaksi'] = data['waktu_pesanan_dibuat'].dt.strftime('%d/%m/%Y')
    data= data.groupby(["tanggal_transaksi", "no_pesanan", "nama_produk", "pesanan_selesai"]).agg(count=("nama_produk", "count"))
    data.reset_index(inplace=True)

    #stopword removal
    def remove_stopwords(text):
        # Define stopwords lists for both languages
        indo_stopwords = set(stopwords.words("indonesian"))
        english_stopwords = set(stopwords.words("english"))
        # Combine stopwords lists (optional, see note below)
        combined_stopwords = indo_stopwords.union(english_stopwords)  # Combine for efficiency
        # Define units to remove
        units = ["ml", "kg", "kilo", "g", "gr", "gram", "cc", "pcs", "mm", "cm", "l", "liter"]
        # Tokenize the text (lowercase for case-insensitive stopword removal)
        tokens = word_tokenize(text.lower())
        # Remove stopwords, numeric variations, and units
        removed = []
        prev_word = ""
        for word in tokens:
            # Check for unit appended to a word
            if re.match(r"^[0-9.]+$", prev_word) and word in units:
                continue  # Skip the unit if it follows a number

            # Filter out stopwords, numeric variations, and standalone units
            if (
                word not in combined_stopwords
                and not word.isdigit()
                and not re.match(r"^[0-9.]+"+"|".join(units)+"$", word)
            ):
                removed.append(word)
            # Update previous word
            prev_word = word
        return removed
    # Apply the function to the DataFrame
    data["nama_produk_stopword"] = data["nama_produk"].apply(remove_stopwords)

    def remove_stopwords2(text):
        # Define stopwords lists for both languages
        indo_stopwords = set(stopwords.words("indonesian"))
        english_stopwords = set(stopwords.words("english"))
        # Combine stopwords lists (optional, see note below)
        combined_stopwords = indo_stopwords.union(english_stopwords)  # Combine for efficiency
        # Define units to remove
        units = ["ml", "kg", "kilo", "g", "gr", "gram", "cc", "pcs", "mm", "cm", "l", "liter"]
        # Tokenize the text (lowercase for case-insensitive stopword removal)
        tokens = word_tokenize(text.lower())
        # Remove stopwords, numeric variations, and units
        removed = []
        prev_word = ""
        for word in tokens:
            # Check for unit appended to a word
            if re.match(r"^[0-9.]+$", prev_word) and word in units:
                continue  # Skip the unit if it follows a number
            # Filter out stopwords, numeric variations, and standalone units
            if (
                word not in combined_stopwords
                and not word.isdigit()
                and not re.match(r"^[0-9.]+"+"|".join(units)+"$", word)
            ):
                removed.append(word)
            # Update previous word
            prev_word = word
        new_stopword = " ".join(removed)
        return new_stopword

    #feature extraction (TF-IDF)
    def calc_TF(document):
        # Counts the number of times the word appears in review
        TF_dict = {}
        for term in document:
            if term in TF_dict:
                TF_dict[term] += 1
            else:
                TF_dict[term] = 1
        # Computes tf for each word
        for term in TF_dict:
            TF_dict[term] = TF_dict[term] / len(document)
        return TF_dict
    data["TF_dict"] = data["nama_produk_stopword"].apply(calc_TF)

    def calc_DF(tfDict):
        count_DF = {}
        # Run through each document's tf dictionary and increment countDict's (term, doc) pair
        for document in tfDict:
            for term in document:
                if term in count_DF:
                    count_DF[term] += 1
                else:
                    count_DF[term] = 1
        return count_DF
    DF = calc_DF(data["TF_dict"])

    n_document = len(data)

    def calc_IDF(__n_document, __DF):
        IDF_Dict = {}
        for term in __DF:
            IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
        return IDF_Dict
    # Stores the idf dictionary
    IDF = calc_IDF(n_document, DF)

    # calc TF-IDF
    def calc_TF_IDF(TF):
        TF_IDF_Dict = {}
        # For each word in the review, we multiply its tf and its idf.
        for key in TF:
            TF_IDF_Dict[key] = TF[key] * IDF[key]
        return TF_IDF_Dict
    # Stores the TF-IDF Series
    data["TF-IDF_dict"] = data["TF_dict"].apply(calc_TF_IDF)

    data["nama_produk_stopword"] = data["nama_produk"].apply(remove_stopwords2)

    # Menyisipkan code untuk menghitung jumlah nama_produk_stopword per pesanan
    pesanan_counts = data.groupby('no_pesanan')['nama_produk_stopword'].count()
    valid_pesanan = pesanan_counts[pesanan_counts > 1].index
    data = data[data['no_pesanan'].isin(valid_pesanan)]

    data.to_excel("for_apriori.xlsx", index=False)
    return data

# def perform_clustering(file_path, text_column, use_pca=False, n_components=2):
#     # Data Loading
#     data = pd.read_excel(file_path)

#     # TF-IDF Feature Extraction
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(data[text_column])
#     X_dense = X.toarray()

#     # Dimensionality Reduction (optional)
#     if use_pca:
#         pca = PCA(n_components=n_components)
#         X_dense = pca.fit_transform(X_dense)
#         print(f"Explained variance by PCA components: {pca.explained_variance_ratio_}")

#     # Find Optimal Clusters using Silhouette Score
#     best_score = -1
#     optimal_clusters = 0
#     silhouette_scores = []
#     n_samples = X_dense.shape[0]

#     for n_clusters in range(2, n_samples):
#         model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
#         cluster_labels = model.fit_predict(X_dense)

#         silhouette_avg = silhouette_score(X_dense, cluster_labels)
#         silhouette_scores.append(silhouette_avg)

#         if silhouette_avg > best_score:
#             best_score = silhouette_avg
#             optimal_clusters = n_clusters

#     model_cluster = AgglomerativeClustering(n_clusters=optimal_clusters)
#     cluster_labels = model_cluster.fit_predict(X_dense)

#     data['cluster'] = cluster_labels

#     output_filename = 'clustering_results.xlsx'
#     data.to_excel(output_filename, index=False)

def visualize_frequent_products(data):
    count = data['nama_produk_stopword'].value_counts().reset_index()
    count.columns = ['nama_produk_stopword', 'count']
    
    plt.figure(figsize=(20, 15))
    ax = sns.barplot(x="nama_produk_stopword", y="count", data=count, order=count.sort_values('count', ascending=False)['nama_produk_stopword'].head(20))
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
    
    plt.xticks(rotation=-90, fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title("Frequently Purchased Products", fontsize=20)
    plt.xlabel("Product Name", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    
    plt.tight_layout()  # Pastikan seluruh elemen grafik terlihat
    plt.subplots_adjust(bottom=0.5)  # Tambahkan margin di bawah jika diperlukan

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    
    return img_str

# def run_apriori(preprocessed_file):
#     df = pd.read_excel(preprocessed_file)
#     itemsets = df.groupby(['no_pesanan'])['nama_produk_stopword'].apply(list).tolist()
#     te = TransactionEncoder()
#     te_ary = te.fit(itemsets).transform(itemsets)
#     df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
#     # Cek apakah ada frequent itemsets
#     frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
#     if frequent_itemsets.empty:
#         return "Tidak ada rekomendasi bundling produk"
    
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

#     # rules['antecedents_products'] = extract_products_from_frozenset(rules['antecedents'])
#     # rules['consequents_products'] = extract_products_from_frozenset(rules['consequents'])
#     # output_file = "association_rules_with_products.xlsx"
#     # rules.to_excel(output_file, index=False)
#     # return output_file

#     # Sort rules by confidence or any other metric you prefer
#     rules = rules.sort_values(by='confidence', ascending=False)

#     # Select top 12 rules
#     top_12_rules = rules.head(12)

#     # Update the subsequent code to use `top_12_rules` instead of `rules`
#     top_12_rules['antecedents_products'] = extract_products_from_frozenset(top_12_rules['antecedents'])
#     top_12_rules['consequents_products'] = extract_products_from_frozenset(top_12_rules['consequents'])
#     output_file = "association_rules_with_products.xlsx"
#     top_12_rules.to_excel(output_file, index=False)
#     return output_file

def run_apriori(preprocessed_file):
    df = pd.read_excel(preprocessed_file)
    itemsets = df.groupby(['no_pesanan'])['nama_produk_stopword'].apply(list).tolist()
    
    # Convert all items in itemsets to strings
    itemsets = [[str(item) for item in itemset] for itemset in itemsets]
    
    # # Print itemsets for debugging
    # print("Itemsets:", itemsets)
    
    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Cek apakah ada frequent itemsets
    frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        return "Tidak ada rekomendasi bundling produk"
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

    rules['antecedents_products'] = extract_products_from_frozenset(rules['antecedents'])
    rules['consequents_products'] = extract_products_from_frozenset(rules['consequents'])
    output_file = "association_rules_with_products.xlsx"
    rules.to_excel(output_file, index=False)
    return output_file

    # # Sort rules by confidence or any other metric you prefer
    # rules = rules.sort_values(by='confidence', ascending=False)

    # # Select top 12 rules
    # top_12_rules = rules.head(12)

    # # Update the subsequent code to use `top_12_rules` instead of `rules`
    # top_12_rules['antecedents_products'] = extract_products_from_frozenset(top_12_rules['antecedents'])
    # top_12_rules['consequents_products'] = extract_products_from_frozenset(top_12_rules['consequents'])
    # output_file = "association_rules_with_products.xlsx"
    # top_12_rules.to_excel(output_file, index=False)
    # return output_file


def extract_products_from_frozenset(column):
    return column.apply(lambda x: ', '.join(list(x)))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        data = preprocess_data(file_path)
        img_str = visualize_frequent_products(data)
        bundling_recommendations = run_apriori("for_apriori.xlsx")
        
        return render_template('results.html', img_str=img_str, recommendations=bundling_recommendations)

if __name__ == '__main__':
    app.run(debug=True)