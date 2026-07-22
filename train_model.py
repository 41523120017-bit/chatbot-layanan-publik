import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. KAMUS NORMALISASI TYPO & SINGKATAN
# ==========================================
KAMUS_TYPO = {
    "syrt": "syarat", "sarat": "syarat", "syaratt": "syarat",
    "bkn": "bikin", "bbuat": "buat", "ap": "apa", "aj": "aja",
    "y": "ya", "yg": "yang", "kel": "kelurahan", "klurahan": "kelurahan",
    "mna": "mana", "mn": "mana", "ddatangi": "didatangi", "datengi": "didatangi",
    "dimna": "dimana", "mnta": "minta", "dftr": "daftar", "antrian": "antrean",
    "jm": "jam", "operasional": "jam", "jdwl": "jadwal", "stuju": "setuju",
    "btl": "batal", "ga": "tidak", "gak": "tidak", "enggak": "tidak", "ktpel": "ktp"
}

def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-z\s]', '', text)  # Cleaning
    tokens = text.split()  # Tokenization
    # Normalisasi Typo
    normalized_tokens = [KAMUS_TYPO.get(word, word) for word in tokens]
    return " ".join(normalized_tokens)

# ==========================================
# 2. DATASET MINIMAL 200 DATA (5 INTENT)
# ==========================================
data_syarat = [
    "apa syarat bikin KTP", "dokumen buat KK baru apa aja", "syarat buat KTP rusak",
    "persyaratan akta kelahiran", "berkas untuk pindah domisili", "apa saja dokumen KTP el",
    "persyaratan cetak ulang KTP", "dokumen pendukung pindah datang", "syarat buat kartu keluarga",
    "syrt bkn ktp ap aj y", "sarat bbuat kk baru"
] * 6

data_jam = [
    "jam berapa kelurahan buka", "jadwal pelayanan hari senin", "apakah hari sabtu buka",
    "jam operasional kantor pelayanan", "jam tutup layanan publik", "kantor buka jam berapa",
    "jam kerja dinas dukcapil", "pelayanan Buka sampai jam berapa", "jadwal loket pendaftaran",
    "jm brp klurahan buka", "jdwl pelayanan senin"
] * 6

data_janji = [
    "mau buat janji antrean", "daftar jadwal kedatangan tanggal 25-10-2026", 
    "pesan tiket antrean jam 10:00", "saya mau booking jadwal tanggal 12-12-2026", 
    "minta nomor antrean besok", "daftar antrean online", "mau ambil antrean tanggal 01-11-2026",
    "mnta dftr antrian jm 10", "mau bbuat janji antrian"
] * 6

data_konfirmasi = [
    "ya benar", "tidak jadi", "setuju lanjut", "batal mas", "ya saya yakin", 
    "enggak", "ya", "tidak", "ok setuju", "batal", "proses saja", "jangan",
    "stuju", "btl", "gak jadi"
] * 5

data_lokasi = [
    "kelurahan mana yang saya datengi untuk daerah jakarta", "dimana alamat kantor kelurahan",
    "lokasi kelurahan jakarta mana", "posisi kantor kelurahan dimana", "alamat dinas dukcapil jakarta",
    "kelurahan mana yang harus saya datangi", "harus ke kelurahan mana", "lokasi pelayanan jakarta",
    "klurahan mna yg ddatangi", "kel mna yg ddatangi jakarta"
] * 6

dataset = []
for s in data_syarat: dataset.append({"utterance": s, "intent": "tanya_syarat"})
for j in data_jam: dataset.append({"utterance": j, "intent": "tanya_jam_operasional"})
for jn in data_janji: dataset.append({"utterance": jn, "intent": "buat_janji_temu"})
for k in data_konfirmasi: dataset.append({"utterance": k, "intent": "konfirmasi"})
for l in data_lokasi: dataset.append({"utterance": l, "intent": "tanya_lokasi"})

df = pd.DataFrame(dataset)
df['clean_utterance'] = df['utterance'].apply(preprocess_text)

# ==========================================
# 3. TRAINING & EVALUASI MODEL
# ==========================================
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_utterance'])
y = df['intent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("=== METRIK EVALUASI MODEL (P4) ===")
print("Akurasi Sistem:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'model_intent.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\n[INFO] Model dan TF-IDF berhasil dilatih dan disimpan!")