# -*- coding: utf-8 -*-
"""MLT Recommender System.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GlGfdqqTXEiDF36PhE_MPrgjwRQg3Nk1

# **Proyek Machine Learning Terapan**

*   Nama        : Muhammad Khalish
*   E-mail      : khalish.21muhammad07@gmail.com
*   Dicoding ID : https://www.dicoding.com/users/mkhlst/
*   Topic       : Workout Recommendation

# **Deskripsi proyek**

Proyek ini akan menganalisa dataset anggota gym yang melakukan workout dan membuat model yang dapat rekomendasi workout tertentu kepada user dengan metode `Content Based Filtering (CBF)`, `Colaborative Filtering (CF)`, dan gabungan keduanya (Hybrid).

# **Import library yang dibutuhkan**
"""

!pip install -q kaggle

!pip install surprise

# Commented out IPython magic to ensure Python compatibility.
# Import load data library
import os
import random
import pandas as pd
import numpy as np
import surprise
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import train_test_split
from surprise import Reader, Dataset, SVD
from surprise import accuracy

"""# **Data Understanding**

Merupakan proses untuk memahami informasi dalam data dan menentukan kualitas data

## **Data Loading**

Merupakan proses untuk memuat dataset agar dapat digunakan. Dataset yang digunakan bersumber dari `Kaggle` yakni `gym_members_exercise_dataset.csv` dengan author `Vala Khorasani`. Informasi lebih lanjut dapat dilihat sebagai berikut.

**Informasi Dataset**

| Jenis | Keterangan |
| ------ | ------ |
| Title | Gym Members Exercise Dataset |
| Source | [Kaggle](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) |
| Maintainer | [Vala Khorasani ⚡](https://www.kaggle.com/valakhorasani) |
| License | Apache 2.0 |
| Visibility | Public |
| Tags | Computer Science, Exercise, Data Visualization, Classification, Exploratory Data Analysis |
| Usability | 10.00 |
"""

# Membuat direktori baru bernama kaggle
!rm -rf ~/.kaggle && mkdir ~/.kaggle/

# Menyalin berkas kaggle.json pada direktori aktif saat ini ke direktori kaggle
!mv kaggle.json ~/.kaggle/kaggle.json

# Mengubah permission berkas
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d valakhorasani/gym-members-exercise-dataset
!kaggle datasets download -d bitanianielsen/nutrition-daily-meals-in-diseases-cases

# Ekstrak berkas zip
!unzip /content/gym-members-exercise-dataset.zip
!unzip /content/nutrition-daily-meals-in-diseases-cases.zip

"""## **Exploratory Data Analysis (EDA)**

Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data

### **Pengecekan Dataset**

Pengecekan informasi dari dataset mengenai sampel data, jumlah kolom, nama kolom, jumlah data per kolom, tipe data.
"""

gym = pd.read_csv('/content/gym_members_exercise_tracking.csv')

gym.info()

"""Berdasarkan tabel diatas, terdapat 15 Kolom dan 973 baris data.
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

# **Data Preparation**

Data Preparation merupakan proses untuk mempersiapkan data sebelum dilakukan tahap pembuatan model machine learning. Pada tahap ini dilakukan proses `Feature Engineering`, `Data Cleaning`, dan `Data Splitting`.

## **Data Cleaning**

Data Cleaning merupakan Proses untuk `membersihkan data dari missing value` dan `data duplikat` yang dapat mempengaruhi peforma model *machine learning*

### **Menangani Missing Value**
"""

gym.shape

gym.isnull().sum()

"""### **Pengecekan dan Pembersihan data duplikat**"""

print('Jumlah data duplikat pada user:', gym.duplicated().sum())

gym = gym.drop_duplicates()
print('Jumlah data duplikat pada user:', gym.duplicated().sum())

"""## **Feature Engineering**

Feature Engineering merupakan proses transformasi data dengan `menambah`, `mengubah` dan `menghapus` beberapa fitur agar kualitas data menjadi lebih baik dan dapat diterima oleh model machine learning nantinya.
"""

gym['User_ID'] = range(1, len(gym) + 1)
gym.head()

"""Terdapat beberapa fitur pada data diatas yang tidak relevan seperti `Max_BPM`, `Resting_BPM`, `Water_Intake (liters)`, `Experience_Level`, `Workout_Type`, `BMI`, dan `Fat_Percentage` dan juga ada nama fitur yang perlu kita ubah agar lebih simpel untuk dikenali. maka dari itu, kita perlu menggunakan fungsi `drop()` dan `rename()` untuk melakukan kedua hal itu."""

gym = gym.drop(columns=['Max_BPM','Resting_BPM', 'Water_Intake (liters)', 'Experience_Level', 'Workout_Type', 'BMI', 'Fat_Percentage'])
gym = gym.rename(columns={'user_id':'User_ID', 'date':'Date', 'Weight (kg)':'Weight',
                            'Height (m)':'Height', 'Session_Duration (hours)':'Duration',
                            'Workout_Frequency (days/week)':'Frequency'})

"""Penambahan fitur `BMI` digunakan untuk mengklasifikaskan goal workout yang ingin dicapai user dimana data BMI didapat dari perhitungan berat badan `Weight` dibagi dengan kuadrat tinggi badan `Height`."""

gym['BMI'] = (gym['Weight'] / (gym['Height'] ** 2)).round(2).astype(float)
gym.head()

"""Workout goal diklasifikasikan berdasarkan index BMI dimana jika melebihi indeks normal `18.5 - 25` makan diklasifikasikan dengan penurunan berat badan `Weight Loss`, jika kurang dari indeks normal diklasifikasikan penambahan berat badan `Weight Gain` dan jika berada pada indeks normal diklasifikasikan `Flexibility`."""

def classify_bmi(bmi):
    if bmi >= 25:
        return 'Weight Loss'
    elif 18.5 <= bmi < 25:
        return 'Flexibility'
    else:
        return 'Weight Gain'

# Terapkan fungsi classify_bmi ke setiap baris kolom BMI
gym['Goal'] = gym['BMI'].apply(classify_bmi)

# Lihat hasilnya
gym.head()

"""Selanjutnya, kita memberikan label jenis dan nama workout beserta level pengalaman pengguna dengan memberikan data dictionary yang diinginkan dan sekaligus mengklasifikasikan berdasarkan tujuan dan level pengalamannya user."""

goal_type = {
    'Weight Loss': ['Cardio', 'HIIT', 'Flexibility'],
    'Weight Gain': ['Strength', 'Flexibility'],
    'Flexibility': ['Flexibility']
}
name = {
    'Cardio': ['Running', 'Walking', 'Cycling', 'Marathon'],
    'HIIT': ['Burpess', 'High Knees', 'Jump Squats', 'Mountain Climbing'],
    'Flexibility': ['Yoga', 'Zumba', 'Pilates'],
    'Strength': ['Bench Press', 'Deadlifts', 'Squat', 'Plank', 'Push Up']
}
level = {
    'Cardio': {'Marathon': 3, 'Running': 2, 'Walking': 1, 'Cycling': 2},
    'HIIT': {'Burpess': 3, 'High Knees': 2, 'Jump Squats': 2, 'Mountain Climbing': 3},
    'Flexibility': {'Yoga': 1, 'Zumba': 2, 'Pilates': 3},
    'Strength': {'Bench Press': 3, 'Deadlifts': 3, 'Squat': 2, 'Plank': 1, 'Push Up':1}
}
def get_workout_and_level(row):
    goal = row['Goal']
    workout_types = goal_type.get(goal, [])

    if workout_types:
        workout_type = workout_types[0]
        workout_name = random.choice(name.get(workout_type, []))
        experience_level = level.get(workout_type, {}).get(workout_name, 'Unknown')
    else:
        workout_type = 'Unknown'
        workout_name = 'Unknown'
        experience_level = 'Unknown'

    return pd.Series([workout_type, workout_name, experience_level])


gym[['Workout_Type', 'Workout_Name', 'Experience_Level']] = gym.apply(get_workout_and_level, axis=1)
gym['Workout_ID'] = gym['Workout_Name'].astype('category').cat.codes + 1
print(gym.head())

"""**Encoding Feature**

kita perlu mengubah label `gender` dari tipe data string menjadi numerik dengan menggunakan fungsi `replace()` agar dapat digunakan.
"""

gym['Gender'] = gym['Gender'].replace({'Male': 0, 'Female': 1})
gym.head()

"""## **Data Splitting**

*Data Splitting* merupkan proses pembagian data berdasarkan kebutuhan untuk sistem rekomendasi. pada proses ini, data dibagi menjadi 3 yaitu `Data User` yang berisi ragam informasi mengenai user, `Data Workout` yang berisi ragam informasi mengenai Workout, dan `Data Feedback` sebagai data `rating implisit`. Pada data feedback, fungsi `MinMaxScaler()` digunakan untuk normalisasi data numerik dalam rentang 0 hingga 1 agar didapatkan `aggregate` yang menjadi rata-rata untuk rating implisit dari data ini.

### **Data User**
"""

user = pd.DataFrame({
    'User_ID': gym['User_ID'],
    'Age': gym['Age'],
    'Gender': gym['Gender'],
    'Height': gym['Height'],
    'Weight': gym['Weight'],
    'BMI': gym['BMI'],
    'Goal': gym['Goal'],
})
user = user.dropna()
user.head()

user['Goal'].value_counts()

"""### **Data Workout**"""

fit = pd.DataFrame({
    'Workout_ID': gym['Workout_ID'],
    'Workout_Name': gym['Workout_Name'],
    'Workout_Type': gym['Workout_Type'],
    'Experience_Level': gym['Experience_Level']
})
fit.head()

fit['Workout_Name'].value_counts()

"""### **Data Feedback**"""

fb = pd.DataFrame({
    'User_ID': gym['User_ID'],
    'Workout_ID': gym['Workout_ID'],
    'Workout_Name': gym['Workout_Name'],
    'Workout_Type': gym['Workout_Type'],
    'Duration': gym['Duration'],
    'Calories_Burned': gym['Calories_Burned'],
    'Avg_BPM': gym['Avg_BPM'],
    'Frequency': gym['Frequency'],
    'Level': gym['Experience_Level']
})
fb

fb[['Duration', 'Calories_Burned', 'Avg_BPM', 'Frequency', 'Level']] = MinMaxScaler().fit_transform(fb[['Duration', 'Calories_Burned',
                                                                                                        'Avg_BPM', 'Frequency', 'Level']]).round(2)
fb['Aggregate'] = ((fb['Duration']  + fb['Calories_Burned'] + fb['Avg_BPM'] + fb['Frequency'] + fb['Level'])/5).round(2)
fb['Preference_Vector'] = fb[['Duration', 'Calories_Burned', 'Avg_BPM', 'Frequency', 'Level']].apply(
    lambda row: row.values, axis=1)
final_fb = fb
final_fb

"""# **Model Development**

Pada taap ini, model yang dibangun ada 3 yaitu `Content-Based Filtering` (CBF), `Collaborative Filtering` (CF) dan `Hybrid Recommendation`.

## **Content-Based Filtering**

### **Cosine Similarity**

Pada tahap ini, kita menggunakan fungsi `cosine_similarity(feature)` untuk menghitung kemiripan antar fitur. Dan juga, fitur yang digunakan pada proyek ini yakni `durasi`, `jumlah kalori yang terbakar`, `rata-rata BPM`, `frekuensi` dan `level pengalaman user`.
"""

features = final_fb[['Duration', 'Calories_Burned', 'Avg_BPM', 'Frequency', 'Level']]

sim_scores = cosine_similarity(features)
print(f'Similarity Scores: ', sim_scores.shape)
print(f'-----'*6)
# Tampilkan hasil similarity scores
print("Cosine Similarity antara workout dan preferensi pengguna:\n", sim_scores)

"""### **Recommendation Testing**"""

def recommend_workout(workout_id, similarity_matrix, top_n=5):
    workout_idx = final_fb[final_fb['Workout_ID'] == workout_id].index[0]

    # Hitung similarity scores dengan workout lain
    similarity_scores = list(enumerate(similarity_matrix[workout_idx]))

    # Urutkan berdasarkan skor similarity tertinggi
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Ambil top-n workout (exclude workout itu sendiri)
    recommended_indices = [i[0] for i in sorted_scores[1:top_n+1]]

    # Tampilkan rekomendasi
    recommendations = final_fb.iloc[recommended_indices][['Workout_Name', 'Workout_Type', 'Level']]
    return recommendations

# Contoh penggunaan
recommended = recommend_workout(1, sim_scores, top_n=5)
print(recommended)

"""## **Collaborative Filtering**

pada model CF proyek ini, library yang digunakan yaitu `Surprice` dimana library ini memiliki banyak algoritma bawaan yang optimal untuk collaborative filtering, memudahkan pengolahan dan evaluasi model dengan metrik standar, dan fleksibel dalam menangani berbagai format data dan memungkinkan penggunaan algoritma custom.

### **Train Test Split**
"""

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(final_fb[['User_ID', 'Workout_ID', 'Aggregate']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

"""### **Modelling**

pada tahap ini, algoritma `SVD` digunakan karena cocok dengan fitur rating yang kita miliki berupa nilai aggregat dari beberapa fitur numerik sebagai nilai rating implisit.
"""

model = SVD()
model.fit(trainset)

predict = model.test(testset)
print(f"RMSE: {accuracy.rmse(predict)}")

"""### **Recommendation Testing**"""

# Mengambil semua workout ID unik
user_id = final_fb.User_ID.sample(1).values[0]
all_workout = final_fb['Workout_ID'].unique()

rated_workouts = final_fb[final_fb['User_ID'] == user_id]['Workout_ID'].values
unrated_workouts = [w for w in all_workout if w not in rated_workouts]

# Melakukan prediksi untuk setiap workout ID dengan model
predictions = [model.predict(uid=user_id, iid=workout_id).est for workout_id in unrated_workouts]

# Mengurutkan indeks berdasarkan skor tertinggi
recommendation_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)

# Menampilkan urutan skor dan indeks
print(f'scores: {predictions}')
print(f'indices: {recommendation_indices}')

# Mengambil top 5 rekomendasi berdasarkan skor tertinggi
recommended_workouts = [unrated_workouts[i] for i in recommendation_indices[:10]]
print(f'Recommended Workouts: {recommended_workouts}')

# Membuat DataFrame rekomendasi
recommendations = fb[fb['Workout_ID'].isin(recommended_workouts)].drop_duplicates(subset=['Workout_Name']).head(10)

# Menampilkan hasil
print("Top 10 workout recommendations for user:", {user_id})
print("-----" * 6)
for row in recommendations.itertuples():
    print(f"{row.Workout_Name}: {row.Workout_Type}")

"""## **Hybrid Recommendation**

### **Score Hybrid**
"""

user_id = 1
workout_id = 4

cf_scores = [model.predict(uid=user_id, iid=workout_id).est for workout_id in unrated_workouts]
print(f'CF Scores: ', cf_scores[:10])
print('-----'*15)

workout_idx = final_fb[final_fb['Workout_ID'] == workout_id].index[0]
similarity_scores = sim_scores[workout_idx]
cbf_scores = similarity_scores[:len(cf_scores)]
print(f'CBF Scores: ', cbf_scores[:10])
print('-----'*15)

hybrid_scores = [0.7 * cf + 0.3 * cbf for cf, cbf in zip(cf_scores, cbf_scores)]
print(f'Hybrid Scores: ', hybrid_scores[:10])

"""### **Recommendation Testing**"""

Sorted_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)
recommended_workouts = [all_workout[i] for i in Sorted_indices[:10]]
print(f'Showing recommendation for user:', {user_id})
print(f'-----'*7)
recommendations = fb[fb['Workout_ID'].isin(recommended_workouts)].drop_duplicates(subset=['Workout_Name']).head(10)
for row in recommendations.itertuples():
    print(f"{row.Workout_Name}: {row.Workout_Type}")

"""# **Evaluation**"""

# Prediksi pada train set
train_predictions = model.test(trainset.build_testset())

# Prediksi pada test set
test_predictions = model.test(testset)

# Hitung RMSE untuk train dan test set
train_rmse = accuracy.rmse(train_predictions, verbose=False)
test_rmse = accuracy.rmse(test_predictions, verbose=False)
print(f'RMSE pada Train Set: {train_rmse}')
print(f'RMSE pada Test Set: {test_rmse}')

# Plot RMSE
plt.bar(['Train Set', 'Test Set'], [train_rmse, test_rmse], color=['blue', 'orange'])
plt.title('RMSE pada Train Set dan Test Set')
plt.ylabel('RMSE')
plt.show()