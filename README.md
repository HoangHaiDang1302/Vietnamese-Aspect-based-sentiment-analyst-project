<div align="center">
  <h1>🇻🇳 Vietnamese ABSA <br> (Aspect-Based Sentiment Analysis)</h1>
  <p><b>Phân tích cảm xúc theo khía cạnh cho đánh giá sản phẩm tiếng Việt.</b></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker">
    <img src="https://img.shields.io/badge/Dataset-VLSP%202018-orange" alt="Dataset">
  </p>
</div>

---

## 📖 Giới thiệu

**Vietnamese ABSA** là một hệ thống Trí tuệ Nhân tạo (NLP) toàn diện, được thiết kế để tự động đọc, trích xuất và phân tích cảm xúc các khía cạnh khác nhau trong bình luận đánh giá sản phẩm bằng tiếng Việt.

Việc chuyển đổi từ một đồ án nghiên cứu mô hình sâu (Jupyter Notebooks) thành một **Web Application Production-ready** giúp đánh giá và thử nghiệm mô hình thực tế một cách cực kỳ trực quan dễ dàng.

Ví dụ, với đánh giá: _"Máy đẹp, pin trâu, camera chụp rõ nét nhưng loa hơi nhỏ"_, hệ thống có khả năng nhận diện:

- `Máy` ➡️ **Positive (Tích cực)** 😊
- `pin` ➡️ **Positive (Tích cực)** 😊
- `camera` ➡️ **Positive (Tích cực)** 😊
- `loa` ➡️ **Negative (Tiêu cực)** 😠

## ✨ Điểm nổi bật

- 🧠 **Đa dạng Kiến trúc Mô hình**: Hỗ trợ chuyển đổi nhanh (Plug-and-predict) giữa **BiGRU-CRF** (nhỏ gọn, nhẹ, thân thiện CPU) và **PhoBERT-CRF** (mô hình ngôn ngữ ngôn ngữ Transformer tiếng Việt với độ chính xác cao).
- ⚡ **API Hiệu năng cao**: Cung cấp backend mạnh mẽ với tốc độ phản hồi cực nhanh, xử lý bất đồng bộ nhờ **FastAPI**.
- 🎨 **Giao diện Web Hiện đại**: Trải nghiệm UI/UX hoàn toàn mới với hiệu ứng Glassmorphism. Tự động highlight đoạn văn bản tương ứng với từng cảm xúc trích xuất được.
- 🐳 **Triển khai Không chạm**: Tất cả mọi thứ từ Backend tới Frontend đều đã được đóng gói hoàn chỉnh với **Docker**.
- 📊 **Kiến trúc Clean Architecture**: Hệ thống cấu trúc rõ ràng giữa việc xử lý suy luận mô hình (ML Inference Pipeline) và tiếp nhận Request Web.

---

## 📂 Cấu trúc Dự án

```text
├── backend/                  # FastAPI Backend API
│   ├── app/
│   │   ├── main.py           # Entry point của ứng dụng
│   │   ├── config.py         # Cấu hình môi trường & hằng số
│   │   ├── models/           # Định nghĩa kiến trúc Model ML (BiGRU, PhoBERT)
│   │   ├── schemas/          # Pydantic schemas (Data Validation)
│   │   ├── services/         # Layer xử lý Logic (Prediction, Tokenization)
│   │   ├── routers/          # Định tuyến API
│   │   └── utils/            # Hàm hỗ trợ
│   └── requirements.txt      # Dependencies cho backend
│
├── frontend/                 # Web UI (HTML5, Modern CSS, Vanilla JS)
│   ├── index.html            # Giao diện chính
│   ├── css/style.css         # UI/UX Styling
│   └── js/app.js             # Logic gọi API và Hiển thị biểu đồ
│
├── models/                   # Trọng số mô hình đã huấn luyện (Pre-trained Weights)
│   ├── bigru_crf.pt          # Mô hình BiGRU-CRF (~12MB)
│   ├── phobert_crf.pt        # Mô hình PhoBERT-CRF (~540MB)
│   └── word2vec.model        # Word2Vec Embeddings tiếng Việt
│
├── data/                     # Dữ liệu huấn luyện & kiểm thử
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
│
├── notebooks/                # Jupyter Notebooks nghiên cứu thuật toán
├── training/                 # Script huấn luyện tự động (Train & Evaluate)
├── docs/                     # Tài liệu thiết kế hệ thống
│
├── Dockerfile                # Bản build Image Docker Backend + Frontend
├── docker-compose.yml        # Trình quản lý Multi-container
├── docker_guide.md           # Hướng dẫn riêng cho Docker
└── README.md                 # Tài liệu tổng quan (bạn đang đọc)
```

---

## 👩‍💻 Hướng dẫn Cài đặt & Chạy (Quick Start)

Dự án có thể chạy trực tiếp trên máy của bạn thông qua Docker hoặc cấu hình chay thủ công Local.

### 🐳 Cách 1: Chạy bằng Docker (Khuyến nghị)

Đây là cách thiết lập nhanh và ít lỗi môi trường nhất. _Hỗ trợ tự động thiết lập quyền đọc model lớn._

> Chi tiết sâu hơn về Docker, vui lòng tham khảo [Docker Guide](docker_guide.md).

```bash
# 1. Clone dự án về máy
git clone <url_repo>
cd Vietnamese-Aspect-based-sentiment-analyst-project

# 2. Khởi tạo bằng Docker Compose
docker-compose up --build
```

> Truy cập Web UI tại: **http://localhost:8000**

### 💻 Cách 2: Chạy trực tiếp qua Local (Dành cho Dev)

```bash
# 1. Tạo môi trường ảo và kích hoạt (dùng Python 3.9+)
python -m venv venv
source venv/bin/activate  # Trên Windows dùng: venv\Scripts\activate

# 2. Cài đặt các thư viện cần thiết
pip install -r backend/requirements.txt

# 3. Khởi chạy Server FastAPI
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

> Giao diện Web: **http://localhost:8000**
>
> Giao diện Tài liệu API Swagger UI: **http://localhost:8000/docs**

---

## 📡 Tài liệu API (Endpoints)

Ứng dụng tuân chuẩn RESTful API với tài liệu tự động:

| Method | Endpoint       | Định dạng Body / Query                  | Chức năng (Description)                                  |
| ------ | -------------- | --------------------------------------- | -------------------------------------------------------- |
| `GET`  | `/api/health`  | -                                       | Kiểm tra trạng thái máy chủ (Health Check).              |
| `GET`  | `/api/models`  | -                                       | Trả về metadata và danh sách các AI Model khả dụng.      |
| `POST` | `/api/predict` | `{"text": "...", "model": "bigru_crf"}` | API cốt lõi: Gửi văn bản đánh giá để bóc tách thông tin. |

**Ví dụ cURL Request Phân tích:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Máy xịn, màn hình rực rỡ nhưng chơi game hơi nóng", "model": "bigru_crf"}'
```

---

## 🏗️ Thông tin Mô hình

Hai mô hình Deep Learning/NLP được tinh chỉnh cấu hình sẵn để suy luận:

| Model           | Kích thước | Micro F1 | Macro F1 | Đặc điểm nổi bật                                                                       |
| --------------- | ---------- | -------- | -------- | -------------------------------------------------------------------------------------- |
| **BiGRU-CRF**   | ~12MB      | 0.7966   | 0.5918   | Kiến trúc chuẩn RNN, rất nhẹ, chạy mượt mà trên môi trường CPU tiêu chuẩn.             |
| **PhoBERT-CRF** | ~540MB     | _-_      | _-_      | Backbones Transformer tiếng Việt cực mạnh, nắm bắt ngữ cảnh tốt (Khuyên dùng với GPU). |

---

## 📊 Bộ dữ liệu (Dataset)

Dự án được Fine-tune và huấn luyện dựa trên bộ dữ liệu nổi tiếng **UIT-ViSD4SA**. Bộ dataset chuyên dụng phân tích đánh giá trên nền tảng thương mại điện tử (domain smartphone):

- **Tập Huấn luyện (Train):** 7,785 câu đánh giá.
- **Tập Validate (Dev):** 1,112 câu đánh giá.
- **Tập Kiểm thử (Test):** 2,225 câu đánh giá.
- **Hệ thống Nhãn (Label Setup):** Được thiết kế với chuẩn **BIO** tags, bao gồm 10 loại khía cạnh (Screen, Camera, Features, Battery, Price, v.v...) × 3 loại Cảm xúc (Positive, Negative, Neutral) tạo ra mạng lưới bóc tách chi tiết sâu sắc.

---

## 🛡️ Giấy phép & Đóng góp

- **Bộ dữ liệu**: Tài sản thuộc về Trường Đại học Công nghệ, Đại học Quốc Gia Hà Nội & nhóm thực hiện nghiên cứu bộ dữ liệu UIT-ViSD4SA. Việc sử dụng tuân thủ tính chất nghiên cứu học thuật.
- **Mã nguồn**: Dự án hoàn toàn nguồn mở. Rất hân hạnh nhận được _Pull Request/Issue_ của các bạn để giúp ứng dụng tối ưu hơn!

---
