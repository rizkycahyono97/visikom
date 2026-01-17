# Tugas Besar 1: Scale-Invariant Feature Transform (SIFT)

**Mata Kuliah: Visi Komputer (T1646746)**

Proyek ini adalah implementasi algoritma SIFT (_Scale-Invariant Feature Transform_) menggunakan Python dan OpenCV untuk mendeteksi fitur, pencocokan fitur (_feature matching_), dan pembuatan panorama sederhana.

## 📋 Deskripsi Tugas

Sesuai dengan instruksi tugas, proyek ini mencakup tahapan:

1.  **Bagian A:** Deteksi dan Visualisasi Keypoint (Lingkaran & Orientasi).
2.  **Bagian B:** Visualisasi Konsep _Difference of Gaussian_ (DoG) dan _Scale Space_.
3.  **Bagian C:** _Feature Matching_ menggunakan BFMatcher dan Lowe's Ratio Test.
4.  **Bagian D:** Pembuatan Panorama (_Image Stitching_) menggunakan Homography dan RANSAC.

## 🛠️ Prasyarat (Requirements)

Pastikan Anda telah menginstal Python 3.x. Library yang digunakan:

- OpenCV (opencv-python)
- NumPy
- Matplotlib

## 🚀 Cara Instalasi & Menjalankan

1.  **Clone atau Download repository ini.**
2.  **Buat dan aktifkan Virtual Environment (Disarankan):**
    ```bash
    git clone https://github.com/rizkycahyono97/visikom
    cd Tugas Besar 1 - Scale-Invariant Feature Transform (SIFT)
    python -m venv venv
    . ./venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Siapkan Gambar:**
    Pastikan ada dua gambar objek yang memiliki area _overlap_ di dalam folder `images/`.
    - Default: `images/jalan1.jpg` dan `images/jalan2.jpg`
    - Anda dapat mengubah nama file di dalam `main.py` variabel `file_gambar_1` dan `file_gambar_2`.
5.  **Jalankan Program:**
    ```bash
    python main.py
    ```
