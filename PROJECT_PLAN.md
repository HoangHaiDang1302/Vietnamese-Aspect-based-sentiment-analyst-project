# 📚 KẾ HOẠCH ĐỀ TÀI DEEP LEARNING

## Aspect-Based Sentiment Analysis cho tiếng Việt với Qwen2.5

**Dataset:** UIT-ViSD4SA  
**Model chính:** Qwen2.5-1.5B-Instruct (Finetuned)  
**Ngày tạo:** 15/12/2024

---

## 📋 MỤC LỤC

1. [Tổng quan đề tài](#1-tổng-quan-đề-tài)
2. [Baseline Models](#2-baseline-models-để-so-sánh)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Ablation Studies](#4-ablation-studies)
5. [Visualizations](#5-visualizations-cần-có)
6. [Cấu trúc báo cáo](#6-cấu-trúc-báo-cáoslides)
7. [Demo Application](#7-demo-application)
8. [Checklist hoàn thiện](#8-checklist-hoàn-thiện-đề-tài)
9. [Timeline đề xuất](#9-timeline-đề-xuất)

---

## 1. TỔNG QUAN ĐỀ TÀI

### 1.1 Bài toán

- **Task:** Aspect-Based Sentiment Analysis (ABSA)
- **Ngôn ngữ:** Tiếng Việt
- **Domain:** Đánh giá sản phẩm (Product Reviews)

### 1.2 Dataset: UIT-ViSD4SA

| Split     | Số lượng samples |
| --------- | ---------------- |
| Train     | 7,785            |
| Dev       | 1,112            |
| Test      | 2,225            |
| **Total** | **11,122**       |

### 1.3 Các Aspect (Khía cạnh)

- GENERAL: Đánh giá chung
- BATTERY: Pin
- CAMERA: Camera/Máy ảnh
- DESIGN: Thiết kế
- FEATURES: Tính năng
- PERFORMANCE: Hiệu năng
- PRICE: Giá cả
- SCREEN: Màn hình
- SER&ACC: Dịch vụ và phụ kiện
- STORAGE: Bộ nhớ

### 1.4 Các Sentiment (Cảm xúc)

- POSITIVE: Tích cực
- NEGATIVE: Tiêu cực
- NEUTRAL: Trung lập

---

## 2. BASELINE MODELS ĐỂ SO SÁNH

### 2.1 Deep Learning Models (RNN-based)

| Model      | Embedding         | Mục đích                   |
| ---------- | ----------------- | -------------------------- |
| **LSTM**   | Word2Vec/FastText | Baseline sequence modeling |
| **BiLSTM** | Word2Vec/FastText | Bidirectional context      |
| **GRU**    | Word2Vec/FastText | Lightweight RNN            |

### 2.2 Transformer-based Models

| Model       | Loại            | Mục đích                     |
| ----------- | --------------- | ---------------------------- |
| **PhoBERT** | Vietnamese BERT | SOTA baseline cho tiếng Việt |

### 2.3 Large Language Models

| Model                        | Loại          | Mục đích        |
| ---------------------------- | ------------- | --------------- |
| **Qwen2.5-1.5B (Finetuned)** | LLM finetuned | **Model chính** |

### 2.5 Kế hoạch thực hiện Baselin

```
Week 1: Deep Learning (LSTM, BiLSTM, GRU)
Week 2: PhoBERT baseline
Week 3: Qwen2.5 finetuning
```

---

## 3. EVALUATION METRICS

### 3.1 Aspect Extraction

- **Precision:** Tỷ lệ aspect dự đoán đúng / tổng aspect dự đoán
- **Recall:** Tỷ lệ aspect dự đoán đúng / tổng aspect thực tế
- **F1-score:** Harmonic mean của Precision và Recall
- **Per-aspect F1:** F1 riêng cho từng aspect (BATTERY, CAMERA, ...)

### 3.2 Sentiment Classification

- **Accuracy:** Tỷ lệ dự đoán đúng
- **Macro F1:** Trung bình F1 của các class
- **Micro F1:** F1 tính trên toàn bộ samples
- **Per-class F1:** F1 riêng cho POSITIVE, NEGATIVE, NEUTRAL

### 3.3 Combined (Aspect + Sentiment)

- **Exact Match:** Đúng cả aspect lẫn sentiment
- **Aspect-Sentiment F1:** F1 cho cặp (Aspect, Sentiment)

### 3.4 Confusion Matrix

- Confusion matrix cho Aspect Extraction
- Confusion matrix cho Sentiment Classification
- Confusion matrix cho (Aspect, Sentiment) pairs

---

## 4. ABLATION STUDIES

### 4.1 Data Size

| % Data | Train samples | Mục đích             |
| ------ | ------------- | -------------------- |
| 25%    | ~1,946        | Low-resource setting |
| 50%    | ~3,892        | Medium setting       |
| 75%    | ~5,839        | Near-full setting    |
| 100%   | 7,785         | Full data            |

### 4.2 LoRA Hyperparameters

| Parameter      | Values to test                 |
| -------------- | ------------------------------ |
| LoRA rank (r)  | 8, 16, 32, 64                  |
| LoRA alpha     | 16, 32, 64                     |
| LoRA dropout   | 0.0, 0.05, 0.1                 |
| Target modules | q_proj, k_proj, v_proj, o_proj |

### 4.3 Training Hyperparameters

| Parameter     | Values to test   |
| ------------- | ---------------- |
| Learning rate | 1e-5, 5e-5, 1e-4 |
| Batch size    | 4, 8, 16         |
| Epochs        | 1, 2, 3, 5       |
| Warmup ratio  | 0.0, 0.03, 0.1   |

### 4.4 Prompt Engineering

| Prompt Type      | Mô tả                          |
| ---------------- | ------------------------------ |
| Simple           | Chỉ yêu cầu phân tích          |
| Detailed         | Có giải thích chi tiết về task |
| With examples    | Có ví dụ trong prompt          |
| Chain-of-thought | Yêu cầu suy luận từng bước     |

### 4.5 Output Format

| Format           | Mô tả             |
| ---------------- | ----------------- |
| JSON             | Structured output |
| Natural Language | Text tự nhiên     |
| Hybrid           | Kết hợp cả hai    |

---

## 5. VISUALIZATIONS CẦN CÓ

### 5.1 Data Analysis

- [ ] Phân bố số lượng samples theo Aspect (Bar chart)
- [ ] Phân bố số lượng samples theo Sentiment (Bar chart/Pie chart)
- [ ] Phân bố Sentiment theo từng Aspect (Stacked bar chart)
- [ ] Phân bố độ dài review (Histogram)
- [ ] Word Cloud cho toàn bộ data
- [ ] Word Cloud cho từng Aspect
- [ ] Word Cloud cho từng Sentiment

### 5.2 Training Visualization

- [ ] Training Loss curve
- [ ] Validation Loss curve
- [ ] Training + Validation Loss (cùng chart)
- [ ] Learning rate schedule (nếu có)

### 5.3 Results Visualization

- [ ] Model comparison bar chart (F1-scores)
- [ ] Per-aspect F1 bar chart
- [ ] Per-sentiment F1 bar chart
- [ ] Confusion Matrix (heatmap) cho Aspect
- [ ] Confusion Matrix (heatmap) cho Sentiment
- [ ] Ablation study results (line charts)

### 5.4 Error Analysis

- [ ] Ví dụ các trường hợp dự đoán sai
- [ ] Phân tích nguyên nhân lỗi
- [ ] Distribution of errors by aspect/sentiment

---

## 6. CẤU TRÚC BÁO CÁO/SLIDES

### 6.1 Cấu trúc báo cáo (Report)

```
Chương 1: Giới thiệu (Introduction)
├── 1.1 Đặt vấn đề (Problem Statement)
├── 1.2 Động lực nghiên cứu (Motivation)
├── 1.3 Mục tiêu (Objectives)
└── 1.4 Phạm vi đề tài (Scope)

Chương 2: Cơ sở lý thuyết (Background)
├── 2.1 Sentiment Analysis
├── 2.2 Aspect-Based Sentiment Analysis
├── 2.3 Large Language Models
├── 2.4 Parameter-Efficient Fine-Tuning (LoRA)
└── 2.5 Qwen2.5 Model

Chương 3: Các công trình liên quan (Related Work)
├── 3.1 ABSA research
├── 3.2 Vietnamese NLP
└── 3.3 LLM finetuning for NLP tasks

Chương 4: Dataset (Data)
├── 4.1 Giới thiệu UIT-ViSD4SA
├── 4.2 Thống kê dữ liệu (EDA)
├── 4.3 Tiền xử lý dữ liệu
└── 4.4 Chuyển đổi định dạng cho finetuning

Chương 5: Phương pháp (Methodology)
├── 5.1 Kiến trúc tổng quan
├── 5.2 Baseline models
├── 5.3 Qwen2.5 finetuning approach
├── 5.4 Prompt design
└── 5.5 Training pipeline

Chương 6: Thí nghiệm (Experiments)
├── 6.1 Thiết lập thí nghiệm
├── 6.2 Hyperparameter settings
├── 6.3 Evaluation metrics
└── 6.4 Ablation studies

Chương 7: Kết quả và phân tích (Results)
├── 7.1 Kết quả baseline models
├── 7.2 Kết quả Qwen2.5 finetuned
├── 7.3 So sánh các models
├── 7.4 Ablation study results
└── 7.5 Error analysis

Chương 8: Demo ứng dụng (Application)
├── 8.1 Mô tả ứng dụng
├── 8.2 Kiến trúc hệ thống
└── 8.3 Demo screenshots

Chương 9: Kết luận (Conclusion)
├── 9.1 Tóm tắt kết quả
├── 9.2 Đóng góp chính
├── 9.3 Hạn chế
└── 9.4 Hướng phát triển

Tài liệu tham khảo (References)
Phụ lục (Appendix)
```

### 6.2 Cấu trúc Slides

```
Slide 1: Title slide
Slide 2-3: Giới thiệu & Motivation
Slide 4: Mục tiêu đề tài
Slide 5-6: Dataset overview
Slide 7: EDA highlights
Slide 8-9: Methodology overview
Slide 10: Baseline models
Slide 11-12: Qwen2.5 finetuning approach
Slide 13-15: Experiments & Results
Slide 16: Model comparison
Slide 17: Ablation studies
Slide 18: Error analysis
Slide 19-20: Demo
Slide 21: Conclusion
Slide 22: Q&A
```

---

## 7. DEMO APPLICATION

### 7.1 Công nghệ sử dụng

- **Framework:** Gradio hoặc Streamlit
- **Model serving:** Transformers + PEFT
- **Deployment:** Local hoặc Hugging Face Spaces

### 7.2 Tính năng demo

```
┌──────────────────────────────────────────────────────────────┐
│           VIETNAMESE ABSA DEMO                                │
│           Phân tích cảm xúc khía cạnh tiếng Việt             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  📝 Nhập đánh giá sản phẩm:                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Máy đẹp, pin trâu, camera chụp đẹp nhưng loa hơi nhỏ   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  [🔍 Phân tích]                                              │
│                                                               │
│  📊 Kết quả phân tích:                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 🟢 DESIGN: Tích cực                                     │ │
│  │    └─ "Máy đẹp"                                         │ │
│  │ 🟢 BATTERY: Tích cực                                    │ │
│  │    └─ "pin trâu"                                        │ │
│  │ 🟢 CAMERA: Tích cực                                     │ │
│  │    └─ "camera chụp đẹp"                                 │ │
│  │ 🔴 FEATURES: Tiêu cực                                   │ │
│  │    └─ "loa hơi nhỏ"                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  📈 Thống kê:                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Tích cực: 3 (75%) | Tiêu cực: 1 (25%) | Trung lập: 0   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 7.3 Demo code outline

```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = PeftModel.from_pretrained(base_model, "path/to/lora")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return model, tokenizer

# Inference function
def analyze_sentiment(review_text):
    # Generate response
    # Parse JSON output
    # Return formatted results
    pass

# Gradio interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Nhập đánh giá sản phẩm"),
    outputs=gr.JSON(label="Kết quả phân tích"),
    title="Vietnamese ABSA Demo",
    description="Phân tích cảm xúc khía cạnh cho đánh giá sản phẩm tiếng Việt"
)

demo.launch()
```

---

## 8. CHECKLIST HOÀN THIỆN ĐỀ TÀI

### 8.1 Data Analysis

- [x] Tạo notebook EDA chi tiết ✅ (nb1.ipynb)
- [x] Vẽ biểu đồ phân bố data ✅
- [x] Tạo Word Cloud ✅
- [x] Viết mô tả dataset ✅

### 8.2 Data Preprocessing

- [x] Script chuyển đổi data format ✅ (nb2_data_preprocessing.ipynb)
- [x] Tạo các file train/dev/test ✅ (4 formats: Qwen, Flat CSV, BIO, Classification)
- [x] Kiểm tra data quality ✅

### 8.3 Baseline Models

- [ ] Implement Deep Learning (LSTM, BiLSTM, GRU)
- [ ] Implement PhoBERT baseline
- [ ] Ghi lại kết quả baseline

### 8.4 Main Model (Qwen2.5 Finetuning)

- [ ] Setup training environment
- [ ] Viết training script
- [ ] Hyperparameter tuning
- [ ] Train final model
- [ ] Save model checkpoints

### 8.5 Evaluation

- [x] Viết evaluation script ✅ (nb7_qwen_evaluation.ipynb)
- [x] Tính các metrics ✅ (Aspect F1, Sentiment F1, Pair F1, Exact Match)
- [x] Tạo confusion matrix ✅ (Aspect & Sentiment)
- [x] So sánh với baselines ✅ (LSTM, BiLSTM, GRU, PhoBERT)

### 8.6 Ablation Studies

- [ ] Data size experiments
- [ ] LoRA hyperparameter experiments
- [ ] Prompt engineering experiments
- [ ] Ghi lại tất cả kết quả

### 8.7 Visualization

- [ ] Training curves
- [ ] Results comparison charts
- [ ] Confusion matrices
- [ ] Error analysis examples

### 8.8 Demo Application

- [ ] Implement Gradio/Streamlit app
- [ ] Test với nhiều ví dụ
- [ ] Screenshots cho báo cáo

### 8.9 Documentation

- [ ] Viết README
- [ ] Comment code đầy đủ
- [ ] Viết báo cáo/paper
- [ ] Tạo slides thuyết trình

### 8.10 Final Review

- [ ] Review code
- [ ] Review báo cáo
- [ ] Kiểm tra reproducibility
- [ ] Chuẩn bị Q&A

---

## 9. TIMELINE ĐỀ XUẤT

### Tuần 1-2: Data & EDA

- [ ] Phân tích dataset
- [ ] Tiền xử lý dữ liệu
- [ ] Chuyển đổi format
- [ ] Tạo visualizations

### Tuần 3: Baseline Models

- [ ] Implement Deep Learning (LSTM, BiLSTM, GRU, TextCNN)
- [ ] Implement PhoBERT
- [ ] Qwen2.5 zero-shot/few-shot
- [ ] Ghi lại kết quả

### Tuần 4-5: Qwen2.5 Finetuning

- [ ] Setup environment
- [ ] Training experiments
- [ ] Hyperparameter tuning
- [ ] Lưu best model

### Tuần 6: Ablation Studies

- [ ] Data size experiments
- [ ] LoRA experiments
- [ ] Prompt experiments
- [ ] Analysis

### Tuần 7: Evaluation & Analysis

- [ ] Final evaluation
- [ ] Error analysis
- [ ] So sánh models
- [ ] Visualizations

### Tuần 8: Demo & Documentation

- [ ] Implement demo app
- [ ] Viết báo cáo
- [ ] Tạo slides
- [ ] Review & polish

---

## 10. TÀI NGUYÊN HỮU ÍCH

### 10.1 Repositories

- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Unsloth: https://github.com/unslothai/unsloth
- PEFT: https://github.com/huggingface/peft
- PhoBERT: https://github.com/VinAIResearch/PhoBERT

### 10.2 Papers

- Qwen2.5 Technical Report
- LoRA: Low-Rank Adaptation of Large Language Models
- PhoBERT: Pre-trained Language Models for Vietnamese

### 10.3 Datasets

- UIT-ViSD4SA (Current dataset)
- UIT-VSFC (Vietnamese feedback analysis)
- ViCoRE (Vietnamese comments)

---

## 📝 NOTES

```
Ghi chú thêm trong quá trình thực hiện:

1. _______________________________________________

2. _______________________________________________

3. _______________________________________________

4. _______________________________________________

5. _______________________________________________
```

---

**Chúc bạn hoàn thành tốt đề tài! 🎓**

_File này được tạo tự động bởi AI Assistant - 15/12/2024_
