# Hệ thống Phát hiện Tin giả (Fake News Detection) 🕵️‍♂️📰

Dự án này là giao diện Web App hỗ trợ việc nhận diện và phân loại tin tức giả (Fake News) hoặc tin thật (True News) bằng việc áp dụng các mô hình Trí tuệ Nhân tạo từ truyền thống đến hiện đại. 

Dự án được phát triển nhằm mục đích phục vụ báo cáo / tiểu luận môn học Xử lý Ngôn ngữ Tự nhiên (NLP).

## 🌟 Các Mô hình được tích hợp
Hệ thống cho phép cắm (plug-and-play) và chạy song song 3 phương pháp tiếp cận khác nhau:
1. **Machine Learning Truyền thống:** `SVM + TF-IDF`
2. **Deep Learning Tuần tự:** `Bi-LSTM + Word2Vec`
3. **Mô hình Ngôn ngữ Lớn (LLMs):** `BERT Fine-tuning`

## 🚀 Công nghệ sử dụng
- **Backend:** Python, Flask, PyTorch, TensorFlow/Keras, Scikit-learn, HuggingFace Transformers.
- **Frontend:** HTML5, Vanilla CSS, Vanilla JavaScript.

---

## 📥 Hướng dẫn Cài đặt & Chạy dự án

### Bước 1: Tải mã nguồn
Đầu tiên, bạn clone repository này về máy:
```bash
git clone https://github.com/ThanhTin926/FakeNewsDetectionWeb.git
cd FakeNewsDetectionWeb
```

### Bước 2: Tải Mô hình (Models) ⚠️ QUAN TRỌNG ⚠️
Do giới hạn dung lượng của GitHub, các file mô hình (Models) có kích thước lớn không được lưu trữ trong mã nguồn này.
Bạn **BẮT BUỘC** phải tải các file mô hình từ Google Drive và đặt vào đúng vị trí để ứng dụng có thể hoạt động.

👉 **[BẤM VÀO ĐÂY ĐỂ TẢI FOLDER MODELS TỪ GOOGLE DRIVE](https://drive.google.com/drive/folders/1Mdcc8tJ8U1WLbLZvcCy6Y0eMfOXxrPcP?usp=sharing)**

Sau khi tải về, hãy giải nén và đảm bảo cấu trúc thư mục `models/` của bạn trông giống như sau (đặt thư mục `models/` nằm cùng cấp với file `app.py`):
```text
FakeNewsDetectionWeb/
├── models/
│   ├── svm_tfidf_fake_news_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── best_bilstm_word2vec_model.keras
│   ├── tokenizer_bilstm_word2vec.pkl
│   ├── bilstm_word2vec_config.json
│   └── best_bert_fake_news_model/
│       ├── config.json
│       ├── model.safetensors
│       ├── vocab.txt
│       └── ...
```

### Bước 3: Cài đặt thư viện (Dependencies)
Đảm bảo máy tính của bạn đã cài đặt Python 3.9+. Mở Terminal tại thư mục code và chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi động Server
Chạy lệnh sau để khởi động Backend Flask:
```bash
python app.py
```
Sau khi thấy thông báo Server đang chạy, hãy mở trình duyệt web và truy cập vào địa chỉ:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📁 Cấu trúc Thư mục Code
```text
FakeNewsDetectionWeb/
│
├── app.py                  # Khởi tạo Flask Server và cung cấp API dự đoán
├── model_handlers.py       # Logic tiền xử lý văn bản và nạp mô hình (Strategy Pattern)
├── requirements.txt        # Danh sách các thư viện cần thiết
├── .gitignore              # Bỏ qua các file rác và thư mục model nặng
│
├── models/                 # Chứa các file mô hình (Cần tự tải từ Google Drive vào)
│
├── static/                 # Chứa file tĩnh cho giao diện
│   ├── style.css           
│   └── script.js           
│
└── templates/              # Chứa file giao diện HTML
    └── index.html          
```

## 👨‍💻 Tác giả
- Phát triển bởi: **Thanh Tin**
- Môn học: **Xử lý Ngôn ngữ Tự nhiên**
