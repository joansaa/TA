import pandas as pd
import string
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

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

    data.to_excel("for_apriori.xlsx", index=False)

def run_apriori(preprocessed_file):
    df = pd.read_excel(preprocessed_file)
    itemsets = df.groupby(['no_pesanan'])['nama_produk_stopword'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_trans, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    rules['antecedents_products'] = extract_products_from_frozenset(rules['antecedents'])
    rules['consequents_products'] = extract_products_from_frozenset(rules['consequents'])
    output_file = "association_rules_with_products.xlsx"
    rules.to_excel(output_file, index=False)
    return output_file

def extract_products_from_frozenset(column):
    return column.apply(lambda x: ', '.join(list(x)))