from flask import Flask, flash, render_template, request, redirect, url_for
import models.knn as knn
import pandas as pd
import numpy as np
from models.config import write_k_value

app = Flask(__name__)
app.secret_key = '{rahasia^negara()indonesia}'

@app.route("/", methods=['GET', 'POST'])
def dataset():
    if request.method == 'GET':
        
        status_data = None
        kolom = None
        row = None
        if not knn.check_available_data():
            status_data = "Data Kosong!"
        
        if knn.check_available_data():
            data = knn.read_dataset().iloc[:5]
            kolom = data.columns.to_list()
            row = data.values.tolist()
            del data
        
        return render_template(
            './pages/dataset.html',
            title = "dataset",
            status_data = status_data,
            kolom = kolom,
            row = row            
        )
        
    if request.method == 'POST':
        dataset = request.files['file']
        dataset.save('./static/data/dataset.csv')
        
        # flashdata untuk mengindikasikan bahwa data berhasil tersimpan
        flash('dataset berhasil diunggah!')
        # mengembalikan link kepada halaman dataset
        return redirect(url_for('dataset'))
    
@app.route("/pemrosesan-data", methods=['GET', 'POST'])
def pemrosesan_data():
    data = knn.knn_method()
    
    return render_template(
        './pages/pemrosesan-data.html',
        title = "Pemrosesan Data",
        jarak = data['jarak'].tolist(),
        indeks = data['indeks'].tolist()
    )

@app.route("/evaluasi-data", methods=['GET', 'POST'])
def evaluasi_data():
    if request.method == 'GET':
        
        data = knn.knn_method()
        akurasi = round(data['akurasi']['accuracy'] * 100, 2)
        akurasi_detail = data['akurasi']
        
        for key, value in akurasi_detail.items():
            if isinstance(value, float):
                akurasi_detail[key] = {'value': value}
        
        akurasi2 = pd.DataFrame.from_dict(akurasi_detail, orient='index')
        akurasi2 = akurasi2.fillna('')
        
        return render_template(
            './pages/evaluasi.html',
            title = "Evaluasi Data",
            data = data,
            akurasi = akurasi,
            akurasi2 = akurasi2.to_html(classes='table table-bordered', justify='left')
        )
    
@app.route("/data-uji", methods=['GET', 'POST'])
def data_uji():
    if request.method == 'GET':
        data_uji = knn.knn_method()
        kolom = pd.DataFrame(data_uji['data_uji']).columns.to_list()
        row = pd.DataFrame(data_uji['data_uji']).values.tolist()
        y_pred = data_uji['y_pred']
        del data_uji
        
        return render_template(
            './pages/data-uji.html',
            title = "Data Uji",
            kolom = kolom,
            row = row,
            y_pred = pd.Series(y_pred).value_counts().to_dict()
        )
        
@app.route("/data-latih", methods=['GET', 'POST'])
def data_latih():
    if request.method == 'GET':
        data_latih = knn.knn_method()
        kolom = pd.DataFrame(data_latih['class']).columns.to_list()
        row = pd.DataFrame(data_latih['class']).values.tolist()
        
        # del data_latih
        
        return render_template(
            './pages/data-latih.html',
            title = "Data Latih",
            kolom = kolom,
            row = row,
            data = data_latih
        )
        
@app.route("/klasifikasi", methods=['GET','POST'])
def klasifikasi():
    data_uji = knn.knn_method()
    kolom = pd.DataFrame(data_uji['data_uji']).columns.to_list()
    
    if request.method == 'GET':
        return render_template(
            './pages/klasifikasi.html',
            title = "klasifikasi",
            kolom = kolom
        )
        
    if request.method == 'POST':
        input = []
        for i in request.form:
            if i != 'nilai-k':
                input.append(float(request.form[i]))
                
        k = int(request.form['nilai-k'])
        
        write_k_value(k)
        
        # input = pd.DataFrame([input], columns=kolom)
        array = np.array(input)
        input = array.reshape(1, -1)
        
        knn_pred = knn.prediksi(input)
        nearest_neighbors = knn.find_k_nearest_neighbors(input, k)
        
        tetangga_terdekat = {
            'columns': list(nearest_neighbors.keys()),
            'rows': nearest_neighbors.values.tolist(),
        }
        
        # return f'{nearest_neighbors}'
        return render_template(
            './pages/hasil-klasifikasi.html',
            prediksi = knn_pred,
            tetangga_terdekat = tetangga_terdekat
            
        )
        
@app.errorhandler(404)
def page_not_found(error):
    return 'Page Not Found', 404