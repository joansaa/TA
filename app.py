from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
from preprocessing import preprocess_data, run_apriori, visualize_frequent_products

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './data'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('process_file', filename=file.filename))
    return redirect(url_for('index'))

@app.route('/process/<filename>')

def process_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    preprocess_data(file_path)
    apriori_output_path = run_apriori("for_apriori.xlsx")
    
    if isinstance(apriori_output_path, str) and apriori_output_path == "Tidak ada rekomendasi bundling produk":
        result = apriori_output_path
        img_str = None  # Tidak ada gambar yang dihasilkan
    else:
        apriori_output = pd.read_excel(apriori_output_path)
        result = apriori_output[['antecedents_products', 'consequents_products']].values.tolist()
        df = pd.DataFrame(result, columns=['nama_produk_stopword', 'consequents_products'])  # Ubah kembali ke dataframe
        img_str = visualize_frequent_products(df)
    
    return render_template('index.html', result=result, img_str=img_str)

if __name__ == '__main__':
    app.run(debug=True)
