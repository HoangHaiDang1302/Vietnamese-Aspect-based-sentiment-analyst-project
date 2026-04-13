# 🇻🇳 Vietnamese ABSA — Aspect-Based Sentiment Analysis

Phân tích cảm xúc theo khía cạnh cho đánh giá sản phẩm tiếng Việt.

## 📂 Cấu trúc Project

```
├── backend/                  # FastAPI Backend
│   ├── app/
│   │   ├── main.py          # Entry point
│   │   ├── config.py        # Configuration
│   │   ├── models/          # ML model definitions
│   │   ├── schemas/         # API schemas
│   │   ├── services/        # Business logic
│   │   ├── routers/         # API routes
│   │   └── utils/           # Utilities
│   └── requirements.txt
│
├── frontend/                 # Web UI (HTML/CSS/JS)
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
│
├── models/                   # Pre-trained weights
│   ├── bigru_crf.pt         # BiGRU-CRF (~12MB)
│   └── phobert_crf.pt       # PhoBERT-CRF (~540MB)
│
├── data/                     # VLSP 2018 Dataset
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
│
├── notebooks/                # Research notebooks
├── training/                 # Training scripts
│
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### 1. Cài đặt dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Chạy local

```bash
# Từ thư mục root project:
uvicorn backend.app.main:app --reload --port 8000
```

Mở trình duyệt: **http://localhost:8000**

### 3. Chạy với Docker

```bash
docker-compose up --build
```

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/models` | Danh sách models |
| `POST` | `/api/predict` | Phân tích sentiment |
| `GET` | `/docs` | Swagger UI |

### Ví dụ gọi API

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Máy đẹp, pin trâu, camera chụp rõ nét", "model": "bigru_crf"}'
```

## 🏗️ Models

| Model | Size | Micro F1 | Macro F1 | Ghi chú |
|-------|------|----------|----------|---------|
| BiGRU-CRF | ~12MB | 0.7966 | 0.5918 | Mặc định, CPU-friendly |
| PhoBERT-CRF | ~540MB | — | — | Cần GPU |

## 📊 Dataset

**VLSP 2018 — UIT-ViSD4SA**

- Train: 7,785 samples
- Dev: 1,112 samples  
- Test: 2,225 samples
- 10 aspects × 3 sentiments = 30 label pairs

## 🛠️ Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS
- **ML:** PyTorch + TorchCRF + Transformers
- **Container:** Docker
