import streamlit as st
import joblib
import re
import json
import random
from datetime import datetime

# Load Model & TF-IDF
model = joblib.load('model_intent.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Pelayanan Publik Kelurahan DKI Jakarta", page_icon="🏛️")
st.title("🏛️ Chatbot Pelayanan Terpadu Kelurahan")
st.caption("Sistem Informasi Kependudukan & Antrean Digital DKI Jakarta")

# Initialize Memory State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "state" not in st.session_state:
    st.session_state.state = "NORMAL"
if "temp_slot" not in st.session_state:
    st.session_state.temp_slot = {}

# Kamus Typo untuk Preprocessing Input Pengguna
KAMUS_TYPO = {
    "syrt": "syarat", "sarat": "syarat", "syaratt": "syarat",
    "bkn": "bikin", "bbuat": "buat", "ap": "apa", "aj": "aja",
    "y": "ya", "yg": "yang", "kel": "kelurahan", "klurahan": "kelurahan",
    "mna": "mana", "mn": "mana", "ddatangi": "didatangi", "datengi": "didatangi",
    "dimna": "dimana", "mnta": "minta", "dftr": "daftar", "antrian": "antrean",
    "jm": "jam", "operasional": "jam", "jdwl": "jadwal", "stuju": "setuju",
    "btl": "batal", "ga": "tidak", "gak": "tidak", "enggak": "tidak", "ktpel": "ktp"
}

def clean_input(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    normalized_tokens = [KAMUS_TYPO.get(word, word) for word in tokens]
    return " ".join(normalized_tokens)

# Logging Function (Aman dari file kosong)
def log_conversation(user_input, intent, slots, response):
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "predicted_intent": intent,
        "extracted_slots": slots,
        "bot_response": response
    }
    logs = []
    try:
        with open("chat_log.json", "r") as f:
            content = f.read().strip()
            if content:
                logs = json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
    
    logs.append(log_data)
    with open("chat_log.json", "w") as f:
        json.dump(logs, f, indent=4)

# Tampilkan Riwayat Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "qr_url" in msg:
            st.image(msg["qr_url"], width=200, caption="Scan QR Code ini di mesin antrean Kelurahan")

# Input Chat Pengguna
user_input = st.chat_input("Ketik pertanyaan Anda di sini...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    bot_response = ""
    qr_code_url = None
    extracted_slots = {}
    predicted_intent = "N/A"

    # ==========================================
    # DIALOG MANAGER & LOGIKA INTENT
    # ==========================================
    
    # State A: Menunggu Konfirmasi Janji Temu
    if st.session_state.state == "WAITING_CONFIRMATION":
        cleaned = clean_input(user_input)
        if any(word in cleaned for word in ["ya", "benar", "setuju", "yakin", "ok"]):
            tanggal = st.session_state.temp_slot.get("tanggal", "terpilih")
            
            # Generasi Nomor Antrean & QR Code Digital
            nomor_antrean = f"A-{random.randint(100, 999)}"
            kode_tiket = f"JAKARTA-KELURAHAN-{nomor_antrean}-{tanggal}"
            qr_code_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={kode_tiket}"
            
            bot_response = (
                f"✅ **ANTREAN BERHASIL TERDAFTAR!**\n\n"
                f"📍 **Lokasi:** Kantor Kelurahan Domisili Anda (DKI Jakarta)\n"
                f"🔢 **Nomor Antrean:** `{nomor_antrean}`\n"
                f"📅 **Tanggal Kedatangan:** {tanggal}\n"
                f"⏰ **Jam Layanan:** 08:00 - 11:00 WIB\n\n"
                f"*Tunjukkan QR Code di bawah ini kepada petugas loket saat Anda tiba:*"
            )
            st.session_state.state = "NORMAL"
            st.session_state.temp_slot = {}
            
        elif any(word in cleaned for word in ["tidak", "batal", "enggak", "jangan"]):
            bot_response = "❌ Pembuatan janji antrean dibatalkan. Ada yang bisa saya bantu lainnya?"
            st.session_state.state = "NORMAL"
            st.session_state.temp_slot = {}
        else:
            bot_response = "Mohon konfirmasi, apakah Anda yakin? Jawab **Ya** untuk setuju atau **Tidak** untuk membatalkan."

    # State B: Mode Percakapan Normal
    else:
        cleaned_text = clean_input(user_input)
        text_vec = tfidf.transform([cleaned_text])
        predicted_intent = model.predict(text_vec)[0]

        if predicted_intent == "tanya_lokasi":
            bot_response = (
                "📍 **Informasi Lokasi Pelayanan DKI Jakarta:**\n\n"
                "Sesuai dengan ketentuan Dinas Dukcapil DKI Jakarta, Anda dapat mendatangi **Kantor Kelurahan sesuai alamat domisili yang tertera pada KTP / KK Anda**.\n\n"
                "Pelayanan administrasi kependudukan (KTP-el, KK, Akta) diproses di loket **Pelayanan Terpadu Satu Pintu (PTSP) Kelurahan** domisili Anda."
            )
            
        elif predicted_intent == "tanya_syarat":
            bot_response = "📋 **Persyaratan Layanan Kependudukan:**\n1. Fotokopi Kartu Keluarga (KK)\n2. Surat Pengantar RT/RW setempat\n3. KTP Lama (jika rusak/perpanjangan) / Akta Kelahiran (untuk pemula)"
            
        elif predicted_intent == "tanya_jam_operasional":
            bot_response = "🕒 **Jam Operasional Kantor Kelurahan DKI Jakarta:**\n- Senin s/d Jumat: 08.00 - 15.00 WIB\n- Sabtu, Minggu & Hari Libur Nasional: **Tutup**"
            
        elif predicted_intent == "buat_janji_temu":
            # Slot filling menggunakan Regex untuk mengekstrak angka tanggal
            date_match = re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', user_input)
            if date_match:
                extracted_date = date_match.group(0)
                extracted_slots["tanggal"] = extracted_date
                st.session_state.temp_slot["tanggal"] = extracted_date
                st.session_state.state = "WAITING_CONFIRMATION"
                bot_response = f"📌 Anda mengajukan antrean ke Kelurahan untuk tanggal **{extracted_date}**. Apakah data ini sudah benar? *(Ya / Tidak)*"
            else:
                bot_response = "Sebutkan tanggal kedatangan Anda dengan format angka (Contoh: *'Mau antrean tanggal 25-10-2026'*)."
                
        elif predicted_intent == "konfirmasi":
            bot_response = "Ada yang bisa saya bantu terkait persyaratan layanan, lokasi Kelurahan, atau jadwal antrean?"

    # Render Respon Bot & Gambar QR
    with st.chat_message("assistant"):
        st.write(bot_response)
        if qr_code_url:
            st.image(qr_code_url, width=200, caption="Scan QR Code ini di mesin antrean Kelurahan")

    # Save Message ke Session State Chat History
    msg_data = {"role": "assistant", "content": bot_response}
    if qr_code_url:
        msg_data["qr_url"] = qr_code_url
    st.session_state.messages.append(msg_data)

    # Save to JSON Log
    log_conversation(user_input, predicted_intent, extracted_slots, bot_response)
    # ==========================================
# FITUR REKAP / DOWNLOAD LOG UNTUK ADMIN
# ==========================================
with st.sidebar:
    st.header("📊 Menu Admin / Evaluasi")
    st.caption("Unduh log percakapan pengguna untuk kebutuhan rekap/laporan.")
    
    try:
        with open("chat_log.json", "r") as f:
            log_data = f.read()
            
        st.download_button(
            label="📥 Download Chat Log (JSON)",
            data=log_data,
            file_name="chat_log_online.json",
            mime="application/json"
        )
    except FileNotFoundError:
        st.info("Belum ada log percakapan tersimpan.")