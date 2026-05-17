async function checkNews() {
    const title = document.getElementById('news-title').value.trim();
    const text = document.getElementById('news-text').value.trim();
    const model = document.getElementById('model-select').value;

    if (!text && !title) {
        alert("Vui lòng nhập nội dung bài báo!");
        return;
    }

    // Reset UI
    document.getElementById('result-section').classList.add('hidden');
    document.getElementById('error-section').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('check-btn').disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, text, model })
        });

        const data = await response.json();

        if (!data.success) {
            showError(data.error);
        } else {
            showResult(data);
        }
    } catch (err) {
        showError("Lỗi kết nối đến máy chủ. Vui lòng thử lại!");
    } finally {
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('check-btn').disabled = false;
    }
}

function showResult(data) {
    const resultSec = document.getElementById('result-section');
    resultSec.classList.remove('hidden');
    
    // Đổi màu border dựa trên kết quả
    if (data.label.includes("Tin giả")) {
        resultSec.classList.add('fake-news');
        document.getElementById('res-label').style.color = "#e53e3e";
    } else {
        resultSec.classList.remove('fake-news');
        document.getElementById('res-label').style.color = "#48bb78";
    }

    document.getElementById('res-label').textContent = data.label;
    document.getElementById('res-prob').textContent = data.probability + "%";
    document.getElementById('res-model').textContent = data.model_used;
    document.getElementById('res-time').textContent = data.processing_time + "s";
    document.getElementById('res-clean-text').textContent = data.preprocessed_text;
}

function showError(msg) {
    const errorSec = document.getElementById('error-section');
    errorSec.classList.remove('hidden');
    document.getElementById('error-msg').textContent = msg;
}
