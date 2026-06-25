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

CPU_TIER_OPTIONS = {
    'i7': 'Intel Core i7',
    'i5': 'Intel Core i5',
    'i3': 'Intel Core i3',
    'AMD_High': 'AMD Ryzen / FX (gama alta)',
    'AMD_Low': 'AMD A-Series / E-Series (gama baja)',
    'Budget': 'Intel Celeron / Pentium / Atom (económico)',
    'Otro': 'Otro (Xeon, Core M, etc.)',
}

GPU_TIER_OPTIONS = {
    'Integrated': 'Integrada (Intel HD / UHD Graphics)',
    'Mid': 'Gama media (Nvidia MX / Intel Iris)',
    'Low': 'Gama baja (AMD Radeon)',
    'High': 'Gama alta (Nvidia GTX/RTX, AMD Radeon Pro/RX, Quadro)',
}

SSD_OPTIONS = [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1024]
HDD_OPTIONS = [0, 32, 128, 500, 1024, 2048]

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
        cpu_ghz = float(request.form['cpu_ghz'])
        cpu_tier = request.form['cpu_tier']
        gpu_tier = request.form['gpu_tier']
        ssd_gb = int(request.form['ssd'])
        hdd_gb = int(request.form['hdd'])
        resolution = request.form['resolution']
        touchscreen = 1 if request.form.get('touchscreen') else 0

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

    return render_template('index.html', prediction=prediction,
                            cpu_tiers=CPU_TIER_OPTIONS, gpu_tiers=GPU_TIER_OPTIONS,
                            ssd_options=SSD_OPTIONS, hdd_options=HDD_OPTIONS)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)