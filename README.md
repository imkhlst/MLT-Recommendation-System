# Laporan Proyek Machine Learning - Muhammad Khalish

## Project Overview
Proyek ini membahas tentang **sistem rekomendasi Workout** berdasarkan **rating implisit** seperti durasi, jumlah kalori, rata-rata BMP, frekuensi, dan level/tingkat pengalaman user.

Dalam beberapa tahun terakhir, minat terhadap kebugaran fisik dan kesehatan telah meningkat secara signifikan. Banyak orang mulai mencari cara untuk menjaga kesehatan melalui berbagai jenis olahraga, baik yang dilakukan di pusat kebugaran (gym) maupun di rumah. Namun, satu tantangan utama yang dihadapi oleh para pengguna adalah menentukan jenis workout yang sesuai dengan kondisi tubuh, preferensi, tingkat kebugaran, dan tujuan mereka, seperti menurunkan berat badan, meningkatkan kekuatan, atau meningkatkan fleksibilitas. Sistem rekomendasi dapat memberikan solusi personal kepada pengguna dengan memperhitungkan preferensi dan riwayat aktivitas mereka[[1](https://ieeexplore.ieee.org/abstract/document/1423975)].

<div align='center'>
  
![Image](https://github.com/user-attachments/assets/770d80f9-f86c-4859-aef0-71297cd1e7f6)

Gambar 1. Gym Room

</div>

Sistem rekomendasi workout hadir sebagai solusi untuk membantu pengguna memilih jenis workout yang paling sesuai dengan kebutuhan mereka. Sistem ini menggunakan algoritma machine learning dan pendekatan berbasis data untuk memberikan saran personal kepada pengguna, baik berdasarkan preferensi pribadi, riwayat aktivitas sebelumnya, maupun tingkat kesulitan latihan[[2](https://www.researchgate.net/publication/227268858_Recommender_Systems_Handbook)].

## Bussiness Undestanding
Pengembangan sistem rekomendasi ini memiliki potensi dan manfaat yang menjadi salah satu alat untuk mengambil keputusan bagi seseorang yang sudah atau akan melakukan workout. Misal, Pengguna dapat terbantu dengan pilihan yang direkomendasikan berdasarkan kondisi mereka saat ini.

### Problem Statement
Berikut ini beberapa masalah yang harus diselesaikan.
1. Bagaimana cara melakukan tahap preprocessing data sebelum dimasukkan ke dalam model Machine Learning?
2. Bagaimana menyiapkan data rating yang relevan untuk metode colaborative filtering pada sistem rekomendasi workout?
3. Bagaimana membuat model yang dapat memberikan sistem rekomendasi berdasarkan rating dari user?

### Goals
Berikut tujuan dari proyek ini.
1. Mengetahui cara mengolah data agar dapat digunakan model
2. Mengetahui fitur-fitur yang relevan sebagai rating implisit untuk metode colaborative filtering
3. Mendapatkan model yang dapat memberikan rekomendasi workout kepada user

### Solution Statement
Berdasarkan rumusan masalah dan tujuan proyek ini, terdapat beberapa pendekatan yang dapat dilakukan sebagai berikut.
1. `Tahap Preprocessing`. hal yang dilakukan pada tahap ini adalah
    * Mengubah atau menambahkan data agar kualitas dataset menjadi lebih bagus.
    * Membersihkan dataset dari missing value dan duplikasi.
2. `Modelling`. pada tahap ini model machine learning menggunakan metode Content Based Filtering dan Colaborative Filtering yang kemudian digabung menjadi Hybrid Recommendation

## Data Understanding

<div align='center'>
  
  Tabel 1. Informasi Umum Dataset
| Jenis | Keterangan |
| ------ | ------ |
| Title | Gym Members Exercise Dataset |
| Source | [Kaggle](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) |
| Maintainer | [Vala Khorasani ⚡](https://www.kaggle.com/valakhorasani) |
| License | Apache 2.0 |
| Visibility | Public |
| Tags | Computer Science, Exercise, Data Visualization, Classification, Exploratory Data Analysis |
| Usability | 10.00 |

</div>

### Exploratory Data Analysis (EDA)
Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Berikut hasil EDA dari `Gym Members Exercise Dataset`.

<div align='center'>
  
  Tabel 2. Informasi Dataset
| # | Column | Non-Null Count | Dtype |
| ----- | ----- | ----- | ----- |
| 0  | Age | 973 | non-null | int64 |
| 1  | Gender | 973 | non-null | object |
| 2  | Weight (kg) | 973 | non-null | float64 |
| 3  | Height (m) | 973 | non-null | float64 |
| 4  | Max_BPM | 973 | non-null | int64 |
| 5  | Avg_BPM | 973 | non-null | int64 |
| 6  | Resting_BPM | 973 | non-null | int64 |
| 7  | Session_Duration (hours) | 973 | non-null | float64 |
| 8  | Calories_Burned | 973 | non-null | float64 |
| 9  | Workout_Type | 973 | non-null | object |
| 10 | Fat_Percentage |  973 | non-null | float64 |
| 11 | Water_Intake (liters) |  973 | non-null | float64 |
| 12 | Workout_Frequency (days/week) | 973 | non-null | int64 | 
| 13 | Experience_Level | 973 | non-null | int64 | 
| 14 | BMI | 973 | non-null | float64 |

</div>

Berdasarkan tabel diatas, terdapat 15 Kolom dan 973 baris data.
- `Age` -> Umur gym member
- `Gender` -> Jenis kelamin gym member
- `Weight (kg)` -> Berat (dalam `kg`) gym member
- `Height (m)` -> Tinggi (dalam `m`) gym member
- `Max_BPM` -> maksimal BPM (Beat Per Minute) gym member saat sesi
- `Avg_BPM` -> Rata-rata BPM (Beat Per Minute) gym member saat sesi
- `Resting_BPM` -> BPM (Beat Per Minute) gym member sebelum sesi
- `Session_Duration (hours)` -> Durasi sesi dalam jam
- `Calories_Burned` -> jumlah kalori yang dibakar tiap sesi
- `Workout_Type` -> kategori workout yang dilakukan
- `Fat_Percentage` -> persentase lemak gym member
- `Water_Intake (liters)` -> kebutuhan air harian (dalam `liter`) saat sesi
- `Workout_Frequency (days/week)` -> jumlah sesi workout per minggu
- `Experience_Level` -> tingkat pengalaman member, (1) Beginner, (2) Intermediate, dan (3) Expert
- `BMI` -> Body Mass Index, dihitung dari tinggi dan berat

## Data Preparation
Data Preparation merupakan proses untuk mempersiapkan data sebelum dilakukan tahap pembuatan model machine learning. Pada tahap ini dilakukan proses `Feature Engineering`, `Data Cleaning`, dan `Data Splitting`.

###  Feature Engineering
*Feature engineering* merupakan proses transformasi data dengan `menambah`, `mengubah` dan `menghapus` beberapa fitur agar kualitas data menjadi lebih baik dan dapat diterima oleh model machine learning nantinya.

1. Selection Feature
Pada tahap ini, dilakukan beberapa perubahan seperti data pada kolom `Workout_Type` tidak relevan dengan parameter lainnya sehingga dihapus bersamaan dengan kolom `Max_BPM`, `Resting_BPM`, `Water_Intake (liters)`, `Experience_Level`, `BMI`, dan `Fat_Percentage` menggunakan fungsi `drop()`. Kemudian, penyederhanaan penamaan untuk kolom `Weight (kg)`, `Height (m)`, `Session_Duration (hours)`, dan `Workout_Frequency (days/week)` menggunakan fungsi `rename()`. dan juga ditambahkan data `User_ID` dengan hasil dapat dilihat pada tabel 3.

<div align='center'>
  
  Tabel 3. Data setelah dilakukan *Feature Selection*
|  | Age | Gender | Weight | Height | Avg_BPM | Duration | Calories_Burned | Frequency | User_ID | BMI 
| ----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |----- | ----- | ----- |
| 0 | 56 | Male |  88.3 | 1.71 | 157 | 1.69 | 1313.0 | 4 | 1 | 30.20 |
| 1 | 46 | Female | 74.9 | 1.53 | 151 | 1.30 | 883.0 | 4 | 2 | 32.00 |
| 2 | 32 | Female | 68.1 |1.66 | 122 | 1.11 | 677.0 | 4 | 3 | 24.71 |
| 3 |25 | Male | 53.2 |1.70 | 164 | 0.59 | 532.0 | 3 | 4 | 18.41 |
| 4 |38 | Male | 46.1 | 1.79 | 158 |  0.64 | 556.0 | 3 | 5 | 14.39 |

</div>

Terakhir ditambahkan data `Goal`, `Workout Type`, `Workout_Name` berdasarkan parameter `Calories_Burned`, `Avg_BPM`, `Level Experience` beserta `Workout_ID`. Hasil dapat dilihat pada tabel 4.

<div align='center'>
  
  Tabel 4. Data setelah dilakukan penambahan data `Goal`, `Workout Type`, `Workout_Name` beserta `Workout_ID`,
|  | Age | Gender | Weight | Height | Avg_BPM | Calories_Burned | Duration | Frequency | User_ID | BMI | Goal | Workout_Type | Workout_Name | Experience_Level | Workout_ID |
| ----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |
| 0 | 56 | Male |  88.3 | 1.71 | 157 | 1.69 | 1313.0 | 4 | 1 | 30.20 | Weight Loss | Cardio | Walking | 1 | 10 | 
| 1 | 46 | Female | 74.9 | 1.53 | 151 | 1.30 | 883.0 | 4 | 2 | 32.00 | Weight Loss | Cardio | Marathon | 3 | 4 |
| 2 | 32 | Female | 68.1 |1.66 | 122 | 1.11 | 677.0 | 4 | 3 | 24.71 | Flexibility | Flexibility | Zumba | 2 | 12 | 
| 3 |25 | Male | 53.2 |1.70 | 164 | 0.59 | 532.0 | 3 | 4 | 18.41 | Weight Gain | Strength | Squat | 2 | 9  |
| 4 |38 | Male | 46.1 | 1.79 | 158 |  0.64 | 556.0 | 3 | 5 | 14.39 | Weight Gain | Strength | Push Up | 1 | 7 |

</div>

2. Encoding Feature
Pada tahap ini, data pada fitur `Gender` diubah dari data string menjadi data numerik dengan menggunakan fungsi `replace()`. hasil dapat dilihat pada tabel 5.

<div align='center'>
  
  Tabel 5. Data setelah dilakukan *Encoding Feature*
|  | Age | Gender | Weight | Height | Avg_BPM | Calories_Burned | Duration | Frequency | User_ID | BMI | Goal | Workout_Type | Workout_Name | Experience_Level | Workout_ID |
| ----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |
| 0 | 56 | 0 |  88.3 | 1.71 | 157 | 1.69 | 1313.0 | 4 | 1 | 30.20 | Weight Loss | Cardio | Marathon | 1 | 10 | 
| 1 | 46 | 1 | 74.9 | 1.53 | 151 | 1.30 | 883.0 | 4 | 2 | 32.00 | Weight Loss | Cardio | Walking | 3 | 4 |
| 2 | 32 | 1 | 68.1 |1.66 | 122 | 1.11 | 677.0 | 4 | 3 | 24.71 | Flexibility | Flexibility | Zumba | 2 | 12 | 
| 3 |25 | 0 | 53.2 |1.70 | 164 | 0.59 | 532.0 | 3 | 4 | 18.41 | Weight Gain | Strength | Squat | 2 | 9  |
| 4 |38 | 0 | 46.1 | 1.79 | 158 |  0.64 | 556.0 | 3 | 5 | 14.39 | Weight Gain | Strength | Push Up | 1 | 7 |

</div>

### Data Splitting

*Data Splitting* merupkan proses pembagian data berdasarkan kebutuhan untuk sistem rekomendasi. pada proses ini, data dibagi menjadi 3 yaitu `Data User` yang berisi ragam informasi mengenai user dapat dilihat pada tabel 6, `Data Workout` yang berisi ragam informasi mengenai Workout dapat dilihat pada tabel 7, dan `Data Feedback` sebagai data `rating implisit` dapat dilihat pada tabel 8. 

<div align='center'>
  
  Tabel 6. Data User
|  | User_ID | Age | Gender | Height | Weight | BMI | Goal |
| ----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |
| 0	| 1	| 56	| 0	| 1.71 | 88.3 | 30.20 |	Weight Loss
| 1	| 2	| 46 | 1 | 1.53	| 74.9 | 32.00 | Weight Loss
| 2	| 3	| 32	| 1	| 1.66 | 68.1 |	24.71 |	Flexibility
| 3	| 4	| 25	| 0 | 1.70 | 53.2 | 18.41 |	Weight Gain
| 4	| 5	| 38	| 0 | 1.79 | 46.1 | 14.39 |	Weight Gain

</div>
<div align='center'>
  
  Tabel 7. Data Workout
|  | Workout_ID | Workout_Name | Workout_Type | Experience_Level |
| ----- | ----- | ----- | ----- |----- |
| 0	| 10	| Walking	| Cardio	| 1	|
| 1	| 4	| Marathon	| Cardio	| 3	|
| 2	| 12	| Zumba	| Flexibility	| 2	|
| 3	| 9	| Squat	| Strength | 2 |
| 4	| 7	| Push Up	| Strength	| 1	|

</div>
Pada dataset ini tidak terdapat data rating secara eksplisit seperti penilaian rating 1 hingga 5 terhadap workout yang dilakukan oleh user sehingga perlu data dalam tipe numerik agar dapat digunakan pada metode `Colaborative Filtering`, hal ini yang disebut dengan `data rating secara implisit`. pada dataset terdapat beberapa parameter yang dapat dijadikan data rating seperti `Avg_BPM`, `Calories_Burned`, `Duration`, `Frequency`, dan `Level`. hal ini dapat dilihat pada tabel 6. Parameter diatas telah dinormalisasi menggunakan fungsi `MinMaxScaler` dan rata-rata hasilnya disimpan pada kolom `Aggregate`.

<div align='center'>
  
  Tabel 8. Data Feedback
|  | User_ID | Workout_ID | Workout_Name | Workout_Type | Duration | Calories_Burned | Avg_BPM | Frequency | Level | Aggregate |
| ----- | ----- | ----- | ----- |----- | ----- | ----- | ----- |----- | ----- | ----- |
| 0	| 1	| 10	| Walking	| Cardio	| 0.79	| 0.68	| 0.76	| 0.67	| 1.0	| 0.78	|
| 1	| 2	| 4	| Marathon	| Cardio	| 0.53	| 0.39	| 0.63	| 0.67	| 1.0	| 0.64	|
| 2	| 3	| 12	| Zumba	| Flexibility	| 0.41	| 0.25	| 0.04	| 0.67	| 0.0	| 0.27	|
| 3	| 4	| 9	| Squat	| Strength | 0.06	| 0.15	| 0.90	| 0.33	| 1.0	| 0.49	|
| 4	| 5	| 7	| Push Up	| Strength	| 0.09	| 0.17	| 0.78	| 0.33	| 1.0	| 0.47 |

</div>

## Modelling
Metode yang digunakan pada tahap ini yaitu *Content Based Filtering* (CBF), *Colaborative Filtering* (CF) dan *Hybrid Recommendation* yang merupakan gabungan dari CBF dan CF. Pada metode CBF, algoritma akan memberikan rekomendasi berdasarkan hal yang serupa dengan konten yang disukai oleh pengguna di masa lalu. Sebagai contoh, misalkan ada seorang pengguna yang melakukan pilates, lalu algoritma ini akan memberikan rekomendasi film yang memiliki kategori yang sama, katakanlah Flexibility. Informasi yang didapatkan akan disimpan berdasarkan vektor. Vektor ini berisi kebiasaan pengguna, seperti workout yang disuka dan tidak disuka dan rating yang diberikan. Vektor ini dinamakan vektor profil. Semua informasi disimpan dalam vektor lain disebut sebagai vektor item. Vektor tersebut dikalkuklasikan dengan persamaan cosine similarity berikut:

$$ sim(A, B) = cos(\theta) = \frac{A . B}{||A||||B||} $$

Dimana:

- A, B menyatakan produk titik dari vektor A dan B
- ||A|| mewakili norma Euclidean (magnitude) dari vektor A.
- ||B|| mewakili norma Euclidean (magnitude) dari vektor B.

Sistem rekomendasi dengan model collaborative filtering memiliki kelebihan dengan kemampuannya memberikan rekomendasi yang personal, namun juga memiliki kelemahan untuk memberikan rekomendasi item yang sangat berbeda dari yang telah disukai pengguna. Pada metode rekomendasi collaborative filtering, algoritma berfokus pada pendapat komunitas pengguna. Pada user-based collaborative filtering, algoritma akan melihat kesamaan selera pengguna.

Terdapat beberapa kelebihan dan kekurangan dari kedua metode tersebut. CF membutuhkan banyak feedback dari pengguna agar sistem berfungsi dengan baik. Sementara itu, content based filtering membutuhkan deskripsi item/fitur yang baik. oleh karena itu, diperlukan solusi untuk mengatasi hal tersebut yaitu dengan *Hybrid System*.

Perhitungan kemiripan menggunakan fungsi `cosine_similarity` dengan fitur yang digunakan pada proyek ini yakni `durasi`, `jumlah kalori yang terbakar`, `rata-rata BPM`, `frekuensi` dan `level pengalaman user`. Setelah dilakukan perhitungan, sistem rekomendasi menggunakan metode CBF menghasilkan 5 top rekomendasi berdasarkan tingkat pengalaman (*Level Experience*) seperti yang ditunjukkan pada tabel 9.

<div align='center'>

  Tabel 9. Hasil Rekomendasi metode CBF
|  | Workout_Name | Workout_Type | Level |
| ----- | ----- | ----- | ----- |
| 345 | Marathon | Cardio | 1.0 |
| 444 | Pilates | Flexibility | 1.0 |
| 597 | Pilates | Flexibility | 1.0 |
| 600 | Pilates | Flexibility | 1.0 |
| 546 | Pilates | Flexibility | 1.0 |

</div>

Sistem rekomendasi menggunakan metode CF memerlukan fungsi `train_test_split` untuk membagi data menjadi `trainset` dan `testset` yang kemudian dilatih menggunakan model `SVD` dengan metrik evaluasi `RMSE`. Hasil top 10 rekomendasi sistem rekomendasi dapat dilihat pada tabel 10.

<div align='center'>

  Tabel 10. Hasil Rekomendasi metode CF
| Top 10 workout recommendations for user: {643} |
| ------------------------------ |
| Marathon: Cardio |
| Zumba: Flexibility |
| Squat: Strength |
| Yoga: Flexibility |
| Running: Cardio |
| Bench Press: Strength |
| Deadlifts: Strength |
| Pilates: Flexibility |
| Cycling: Cardio |
| Plank: Strength |

</div>

Sistem rekomendasi *hybrid* membutuhkan `hybrid_score` yang merupakan penjumlahan dari `cf score` dan `cbf score` yang masing-masing dikalikan bobotnya. cbf score didapatkan dengan fungsi `cosine_similarity` dab cf score didapatkan dari hasil prediksi model yang dilatih sebelumnya. Pada proyek ini, masing-masing bobot diatur sebesar `0.7` untuk cf score dan `0.3` untuk cbf score. hasil rekomendasi top 10 rekomendasi sistem hybrid dapat dilihat pada tabel 11.

<div align='center'>

  Tabel 11. Hasil Rekomendasi metode hybrid
| Showing recommendation for user: {1} |
| ----------------------------------- |
| Walking: Cardio |
| Marathon: Cardio |
| Zumba: Flexibility |
| Push Up: Strength |
| Yoga: Flexibility |
| Running: Cardio |
| Bench Press: Strength |
| Deadlifts: Strength |
| Pilates: Flexibility |
| Cycling: Cardio |

</div>

## Evaluation
Pengukuran performa dari model sistem rekomendasi bergantung pada jenis sistem rekomendasi yang digunakan. Untuk model dengan *Content-based Filtering*, performa akan dihitung berdasarkan seberapa cocok produk yang direokmendasikan dengan kategorinya. Sedangkan *Collaborative filtering* akan menggunakan metrik pengukuran *model based*, contohnya RMSE. Untuk pendekatan *hybrid* tidak dilakukan pengecekan karena metode ini gabungan dari keduanya, jika keduanya memiliki hasil yang bagus, maka model *hybrid* akan memberikan rekomendasi bagus pula. 

### Pengujian Model *Content Based Filtering*

Model ini hanya menggunakan metrik *Precision* untuk mengetahui seberapa baik perforam model tersebut. Presisi adalah metrik yang biasa digunakan untuk mengevaluasi kinerja model pengelompokan. Metrik ini menghitung rasio antara nilai *ground truth* (nilai sebenarnya) dengan nilai prediksi yang positf. Perhitungan rasio ini dijabarkan melalui rumus di bawah ini:

$$ Precision = \frac{TP}{TP + FP} $$

Dimana:

- TP (*True Positive*), jumlah kejadian positif yang diprediksi dengan benar.
- FP (*False Positive*), jumlah kejadian positif yang diprediksi dengan salah.

Berdasarkan hasil yang dikeluarkan berdasarkan tabel 5 dapat dilihat bahwasanya besar presisi jika dihitung adalah 5/5 untuk rekomendasi Top-5. Ini menunjukan sistem mampu memberikan rekomendasi sesuai dengan kategorinya.  

### Pengujian Model *Collaborative Filtering*

Evaluasi metrik yang dapat digunakan untuk mengukur kinerja model ini adalah metrik RMSE (*Root Mean Squared Error*). RMSE adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. RMSE dapat dijabarkan melalui pendekatan rumus berikut ini

$$ RMSE =  \sqrt{\frac{\sum_{t=1}^{n}(A_t - F_t)^2}{n}} $$

Dimana:

- $A_t$ : Nilai aktual
- $F_t$ : Nilai hasil prediksi
- n: Banyak data

*Collaborative Filtering* dengan model `SVD` memiliki nilai metrik `RMSE` sebesar `0.1498` untuk testset dan `0.1274` untuk trainset, dapat dilihat pada gambar 2.

<div align='center'>
  
![image](https://github.com/user-attachments/assets/440cbb13-8327-4b47-b052-162c2443ce4a)

Gambar 2. Grafik Evaluasi RMSE

</div>

## Conclusion
Dari hasil evaluasi diatas, dapat disimpulkan:
1. Data rating dapat digantikan oleh parameter lain seperti `Avg_BPM`, `Calories_Burned`, `Duration`, `Frequency`, dan `Level` untuk kasus rekomendasi workout.
2. Model dengan metoda Content Based Filtering memiliki peforma yang baik dengan tingkat presisi `100%` dan metoda Colaborative Filtering memiliki nilai metrik RMSE `0.1242` untuk trainset dan `0.1503` untuk testset.

## References
[1] Adomavicius, G., & Tuzhilin, A. (2005). Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions. IEEE Transactions on Knowledge and Data Engineering.

[2] Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.

[3] Gym Members Exercise Dataset. [Link](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)
