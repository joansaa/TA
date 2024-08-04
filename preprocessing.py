from flask import Flask, request, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
import seaborn as sns
import base64
import os
import textwrap
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

    #normalization
    def normalize_text(data):
    # Memecah string data menjadi daftar kata-kata berdasarkan spasi
    # Menggabungkan kata-kata dalam daftar yang dihasilkan oleh data.split() menjadi satu string, dengan satu spasi di antara setiap kata
        return " ".join(data.split())

    data["nama_produk"] = data["nama_produk"].apply(normalize_text)

    #remove duplicates word
    def remove_duplicates(text):
        words = text.split()
        seen = []
        for word in words:
            if word not in seen:
                seen.append(word)
        return " ".join(seen)

    data["nama_produk"] = data["nama_produk"].apply(remove_duplicates)

    #filtering
    data['pesanan_selesai'] = data[data["status_pesanan"] == "selesai"][["status_pesanan"]]

    # filtering and grouping
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
        units = ["ml", "kg", "kilo", "g", "gr", "gram", "cc", "pcs", "mm", "cm", "l", "liter", "capsul", "kapsul", "botol", "pot" "box" "w"]
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords, numeric variations, and units
        unremoved = []
        for word in tokens:
            # Filter out stopwords, numeric variations, and standalone units
            if (
                # Memeriksa apakah word tidak termasuk dalam daftar combined_stopwords
                word not in combined_stopwords
                # Memeriksa apakah word bukan merupakan angka
                and not word.isdigit()
                # Memeriksa apakah word bukan merupakan variasi numerik yang diikuti oleh unit tertentu
                and not any(re.match(r"^\d+\.?\d*\s*"+unit, word) for unit in units)
            ):
                # Jika word memenuhi semua kondisi dalam pernyataan if (tidak dalam stopwords, bukan angka murni, 
                # dan bukan variasi numerik dengan unit tertentu), maka word ditambahkan ke daftar unremoved.
                unremoved.append(word)
        # Menggabungkan semua kata dalam daftar unremoved menjadi satu string, dengan satu spasi di antara setiap kata
        new_stopword = " ".join(unremoved)
        return new_stopword

    data["nama_produk_stopword"] = data["nama_produk"].apply(remove_stopwords)

    #menghitung jumlah nama_produk_stopword per pesanan
    #hanya mengambil data yang membeli 2 atau lebih produk dalam 1x pesanan
    pesanan_counts = data.groupby('no_pesanan')['nama_produk_stopword'].count()
    valid_pesanan = pesanan_counts[pesanan_counts > 1].index
    data = data[data['no_pesanan'].isin(valid_pesanan)]

    data.to_excel("for_apriori.xlsx", index=False)
    return data

# Function to visualize frequent products
def visualize_frequent_products(data):
    # Count the occurrences of each product
    count = data['nama_produk_stopword'].value_counts().reset_index()
    count.columns = ['nama_produk_stopword', 'count']

    # Mengambil kata pertama dari nama produk
    count['nama_produk_stopword'] = count['nama_produk_stopword'].apply(lambda x: x.split()[0] if x else '')

    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))  # Set the figure size
    ax = sns.barplot(y="nama_produk_stopword", x="count", data=count,
                    order=count.sort_values('count', ascending=False)['nama_produk_stopword'].head(10),
                    palette="viridis", ci=None)  # Use the 'viridis' palette for color variety

    for p in ax.patches:
        ax.annotate(format(p.get_width(), '.0f'),
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='center', va='center',
                    xytext=(5, 0),
                    textcoords='offset points')

    plt.yticks(fontsize=12)
    plt.xticks(range(0, count['count'].max() + 1, 1), fontsize=12)
    plt.ylabel("Nama Produk", fontsize=12, labelpad=20)  # Add padding to the y-axis label
    plt.xlabel("Jumlah Pembelian", fontsize=12, labelpad=20)  # Add padding to the x-axis label

    plt.tight_layout()
    plt.subplots_adjust(left=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()

    return img_str


# def visualize_frequent_products(data):
#     count = data['nama_produk_stopword'].value_counts().reset_index()
#     count.columns = ['nama_produk_stopword', 'count']
    
#     # Mengambil kata pertama dari nama produk
#     count['nama_produk_stopword'] = count['nama_produk_stopword'].apply(lambda x: x.split()[0] if x else '')
    
#     sns.set(style="whitegrid")
#     plt.figure(figsize=(15, 10))  # Mengurangi ukuran gambar
#     ax = sns.barplot(y="nama_produk_stopword", x="count", data=count, 
#                     order=count.sort_values('count', ascending=False)['nama_produk_stopword'].head(10), 
#                     color='#007aff', ci=None)  # Menggunakan satu warna untuk semua bar
    
#     for p in ax.patches:
#         ax.annotate(format(p.get_width(), '.0f'), 
#                     (p.get_width(), p.get_y() + p.get_height() / 2.), 
#                     ha='center', va='center', 
#                     xytext=(5, 0), 
#                     textcoords='offset points')
    
#     plt.yticks(fontsize=12)
#     plt.xticks(range(0, count['count'].max() + 1, 1), fontsize=12)
#     plt.ylabel("Product Name", fontsize=12, labelpad=20)  # Tambahkan padding pada label sumbu y
#     plt.xlabel("Count", fontsize=12, labelpad=20)  # Tambahkan padding pada label sumbu x
    
#     plt.tight_layout()
#     plt.subplots_adjust(left=0.3)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)

#     buffer = BytesIO()
#     plt.savefig(buffer, format="png")
#     buffer.seek(0)
#     img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
#     buffer.close()
    
#     return img_str


def run_apriori(preprocessed_file):
    df = pd.read_excel(preprocessed_file)
    itemsets = df.groupby(['no_pesanan'])['nama_produk_stopword'].apply(list).tolist()
    
    # Convert all items in itemsets to strings
    itemsets = [[str(item) for item in itemset] for itemset in itemsets]
    
    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Cek apakah ada frequent itemsets
    frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        return "Tidak ada frequent itemset yang ditemukan"
    
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