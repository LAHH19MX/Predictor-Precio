from flask import Flask, render_template, request
import joblib
import numpy as np
import re
import os

app = Flask(__name__)

pipeline = joblib.load('pipeline_laptop.pkl')
modelo = pipeline['modelo']
encoder = pipeline['encoder']
scaler_robust = pipeline['scaler_robust']
scaler_standard = pipeline['scaler_standard']
pca = pipeline['pca']

COLS_CAT = ['Company', 'TypeName', 'OpSys', 'Gpu_Tier', 'Cpu_Tier']
COLS_NUM = ['Inches', 'Ram_GB', 'Weight_kg', 'Cpu_GHz', 'SSD_GB', 'HDD_GB', 'Resolution']

def parse_cpu_tier(cpu_text):
    if 'Core i7' in cpu_text: return 'i7'
    if 'Core i5' in cpu_text: return 'i5'
    if 'Core i3' in cpu_text: return 'i3'
    if any(k in cpu_text for k in ['Ryzen', 'FX']): return 'AMD_High'
    if 'AMD' in cpu_text: return 'AMD_Low'
    if any(k in cpu_text for k in ['Celeron', 'Pentium', 'Atom']): return 'Budget'
    return 'Otro'

def parse_gpu_tier(gpu_text):
    if any(k in gpu_text for k in ['GTX', 'RTX', 'Quadro', 'Tesla', 'Titan', 'Radeon Pro', 'Radeon R9', 'Radeon RX', 'FirePro']):
        return 'High'
    if any(k in gpu_text for k in ['Nvidia', 'Iris']):
        return 'Mid'
    if 'AMD' in gpu_text:
        return 'Low'
    return 'Integrated'

def parse_storage(memory_text):
    ssd, hdd = 0, 0
    for parte in memory_text.split('+'):
        m = re.search(r'([\d.]+)(TB|GB)', parte.strip())
        if m:
            val = float(m.group(1))
            if m.group(2) == 'TB':
                val *= 1024
            if 'SSD' in parte or 'Flash' in parte:
                ssd += val
            else:
                hdd += val
    return ssd, hdd

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        company = request.form['company']
        typename = request.form['typename']
        opsys = request.form['opsys']
        inches = float(request.form['inches'])
        ram_gb = int(request.form['ram'])
        weight_kg = float(request.form['weight'])
        cpu_text = request.form['cpu']
        gpu_text = request.form['gpu']
        memory_text = request.form['memory']
        resolution = request.form['resolution']
        touchscreen = 1 if request.form.get('touchscreen') else 0

        cpu_ghz_match = re.search(r'(\d+\.?\d*)GHz', cpu_text)
        cpu_ghz = float(cpu_ghz_match.group(1)) if cpu_ghz_match else 2.5
        cpu_tier = parse_cpu_tier(cpu_text)
        gpu_tier = parse_gpu_tier(gpu_text)
        ssd_gb, hdd_gb = parse_storage(memory_text)

        res_match = re.match(r'(\d+)x(\d+)', resolution)
        res_pixels = int(res_match.group(1)) * int(res_match.group(2)) if res_match else 1920 * 1080

        cat_values = np.array([[company, typename, opsys, gpu_tier, cpu_tier]])
        cat_encoded = encoder.transform(cat_values)

        num_values = np.array([[inches, ram_gb, weight_kg, cpu_ghz, ssd_gb, hdd_gb, res_pixels]])
        num_scaled = scaler_robust.transform(num_values)

        row = np.zeros((1, 13))
        row[0, 0] = cat_encoded[0, 0]  # Company
        row[0, 1] = cat_encoded[0, 1]  # TypeName
        row[0, 2] = num_scaled[0, 0]   # Inches
        row[0, 3] = cat_encoded[0, 2]  # OpSys
        row[0, 4] = num_scaled[0, 1]   # Ram_GB
        row[0, 5] = num_scaled[0, 2]   # Weight_kg
        row[0, 6] = num_scaled[0, 3]   # Cpu_GHz
        row[0, 7] = cat_encoded[0, 4]  # Cpu_Tier
        row[0, 8] = num_scaled[0, 4]   # SSD_GB
        row[0, 9] = num_scaled[0, 5]   # HDD_GB
        row[0, 10] = touchscreen
        row[0, 11] = num_scaled[0, 6]  # Resolution
        row[0, 12] = cat_encoded[0, 3] # Gpu_Tier

        row_std = scaler_standard.transform(row)
        row_pca = pca.transform(row_std)
        pred = modelo.predict(row_pca)[0]
        prediction = round(pred, 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
