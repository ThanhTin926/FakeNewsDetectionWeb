from flask import Flask, request, jsonify, render_template
from model_handlers import SVMPredictor, BiLSTMPredictor, BERTPredictor

app = Flask(__name__)

print("Đang khởi tạo các mô hình...")
models = {
    "svm": SVMPredictor(),
    "bilstm": BiLSTMPredictor(),
    "bert": BERTPredictor()
}

model_status = {}
for name, predictor in models.items():
    model_status[name] = predictor.load()
    status_str = "Thành công" if model_status[name] else "Không tìm thấy file"
    print(f"- Load mô hình {name.upper()}: {status_str}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    title = data.get("title", "")
    text = data.get("text", "")
    model_choice = data.get("model", "svm")

    if not title and not text:
        return jsonify({"success": False, "error": "Vui lòng nhập nội dung tin tức."}), 400

    if model_choice not in models:
        return jsonify({"success": False, "error": "Mô hình không hợp lệ."}), 400

    if not model_status[model_choice]:
        return jsonify({"success": False, "error": f"Mô hình {model_choice.upper()} chưa được load thành công do thiếu file."}), 500

    try:
        predictor = models[model_choice]
        result = predictor.predict(title, text)
        result["model_used"] = model_choice.upper()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
