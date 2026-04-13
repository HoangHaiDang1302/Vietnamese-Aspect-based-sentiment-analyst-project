# 🔬 Phân Tích Không Gian Cải Tiến Các Baseline Models

## 📊 Tình Trạng Hiện Tại

**Best Baseline:** BiGRU-CRF (Micro F1: **79.66%**, Macro F1: **59.18%**)

### Bảng tổng hợp 6 models

| Model | Token Acc | Micro F1 | Macro F1 | Weighted F1 |
|:---|:---:|:---:|:---:|:---:|
| **BiGRU-CRF** | **0.6923** | **0.7966** | **0.5918** | **0.7925** |
| BiLSTM-CRF | 0.6779 | 0.7853 | 0.5850 | 0.7811 |
| TextCNN-CRF | 0.6772 | 0.7858 | 0.5541 | 0.7776 |
| GRU-CRF | 0.6500 | 0.7639 | 0.5348 | 0.7539 |
| LSTM-CRF | 0.6576 | 0.7622 | 0.5160 | 0.7474 |
| RNN-CRF | 0.6515 | 0.7614 | 0.4897 | 0.7406 |

### Kết quả chi tiết best model (BiGRU-CRF)

**Overall (span-based):**

| Sub-task | P_Micro | R_Micro | F1_Micro | P_Macro | R_Macro | F1_Macro |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Aspect | 54.56 | 53.38 | 53.97 | 51.95 | 49.52 | 50.55 |
| Polarity | 53.90 | 52.74 | 53.32 | 43.52 | 41.58 | 42.48 |
| Aspect-Polarity | 51.71 | 50.60 | 51.15 | 38.05 | 35.65 | 36.27 |

**Per-Aspect F1:** BATTERY (62.37) > CAMERA (60.37) > GENERAL (57.67) > DESIGN (53.25) > SCREEN (51.90) > SER&ACC (51.15) > PERFORMANCE (50.88) > FEATURES (45.13) > STORAGE (38.60) > PRICE (34.16)

**Per-Sentiment F1:** POSITIVE (62.87) >> NEGATIVE (39.84) >> NEUTRAL (24.74)

**Aspect#Polarity — Các nhãn CỰC TỆ (F1 ≈ 0):**
- STORAGE#NEUTRAL: **0.00**
- SER&ACC#NEUTRAL: **0.00**
- DESIGN#NEUTRAL: **8.33**
- PERFORMANCE#NEUTRAL: **11.85**
- PRICE#NEUTRAL: **13.33**
- SER&ACC#NEGATIVE: **16.67**

---

## ✅ CÂU TRẢ LỜI: **CÓ, CÒN RẤT NHIỀU KHÔNG GIAN CẢI TIẾN!**

Dựa trên phân tích, tôi chia không gian cải tiến thành **5 nhóm chính**, từ dễ → khó, từ tác động nhỏ → lớn:

---

## 🟢 Nhóm 1: Cải Tiến Embedding (Tác động: ⭐⭐⭐ Trung bình-Cao)

### Hiện trạng
- Sử dụng **Word2Vec tự train** với dimension = 150, trained trên chính dataset (~11K câu)
- Corpus quá nhỏ → embedding chất lượng kém, không capture đủ semantic

### Cải tiến khả thi

| Cải tiến | Dự đoán tác động | Độ khó |
|:---|:---:|:---:|
| **PhoBERT-base embeddings** (freeze) | +5~8% F1 | ⭐⭐ Dễ |
| **FastText pre-trained Vietnamese** (cc.vi.300) | +3~5% F1 | ⭐ Rất dễ |
| **Tăng W2V dim → 300** + train trên corpus ngoài | +2~3% F1 | ⭐ Rất dễ |
| **Character-level + Word-level embedding** kết hợp | +2~4% F1 | ⭐⭐⭐ Trung bình |

> [!TIP]
> Thay Word2Vec 150d bằng **FastText pre-trained Vietnamese 300d** là cải tiến "low-hanging fruit" dễ nhất, chỉ cần thay embedding matrix. FastText xử lý OOV tốt hơn Word2Vec đáng kể cho tiếng Việt.

---

## 🟡 Nhóm 2: Xử Lý Data Imbalance (Tác động: ⭐⭐⭐⭐ Cao)

### Hiện trạng — Đây là vấn đề LỚN NHẤT

Mô hình gần như **không học được gì** cho các nhãn ít mẫu:
- NEUTRAL sentiment chỉ đạt F1 = 24.74% (so với POSITIVE = 62.87%)
- Nhiều nhãn Aspect#Polarity có F1 = 0% (STORAGE#NEUTRAL, SER&ACC#NEUTRAL)
- Khoảng cách Micro F1 và Macro F1 rất lớn (~15-17 điểm) → mô hình bias nặng về majority class

### Cải tiến khả thi

| Cải tiến | Dự đoán tác động | Độ khó |
|:---|:---:|:---:|
| **Class-weighted CRF loss** | +5~10% Macro F1 | ⭐⭐ Dễ |
| **Focal Loss** thay CRF loss thường | +3~7% Macro F1 | ⭐⭐ Dễ-Trung bình |
| **Oversampling** các câu chứa nhãn hiếm | +3~5% Macro F1 | ⭐ Rất dễ |
| **Data Augmentation** (synonym replacement, back-translation) | +5~8% Macro F1 | ⭐⭐⭐ Trung bình |
| **SMOTE trên embedding space** | +2~4% Macro F1 | ⭐⭐⭐ Trung bình |

> [!IMPORTANT]
> **Class-weighted CRF loss** là cải tiến có ROI cao nhất. Hiện CRF loss đang weight bằng nhau cho tất cả tag, dẫn đến model bị bias nặng về tag `O` (chiếm 80%+ token) và các tag majority như `B-*#POSITIVE`. Chỉ cần thêm weight vào loss function là có thể cải thiện Macro F1 đáng kể.

---

## 🟠 Nhóm 3: Cải Tiến Kiến Trúc Model (Tác động: ⭐⭐⭐ Trung bình)

### Hiện trạng
- RNN/LSTM/GRU + CRF là kiến trúc cơ bản, 2 layers, hidden=256
- Không có cơ chế **Attention** → thiếu khả năng focus vào các từ quan trọng
- TextCNN chỉ dùng 3 kernel sizes (3, 5, 7) → limited receptive field

### Cải tiến khả thi

| Cải tiến | Dự đoán tác động | Độ khó |
|:---|:---:|:---:|
| **Self-Attention layer** sau BiGRU/BiLSTM | +3~5% F1 | ⭐⭐ Dễ |
| **Multi-head Attention** + CRF | +4~6% F1 | ⭐⭐⭐ Trung bình |
| **Stacked BiGRU** (3-4 layers thay vì 2) | +1~2% F1 | ⭐ Rất dễ |
| **Dilated CNN** thay thế CNN thường | +2~3% F1 | ⭐⭐ Dễ |
| **Hybrid CNN-BiGRU-CRF** (CNN extract local, RNN extract global) | +3~5% F1 | ⭐⭐⭐ Trung bình |
| **Residual connections** giữa các layers | +1~2% F1 | ⭐⭐ Dễ |

> [!NOTE]
> Thêm **Self-Attention** sau BiGRU là cải tiến kiến trúc hiệu quả nhất. Nó giúp model "nhìn" được mối quan hệ giữa aspect word và sentiment word ở xa nhau trong câu, điều mà RNN thuần túy làm không tốt.

---

## 🔴 Nhóm 4: Cải Tiến Training Strategy (Tác động: ⭐⭐ Trung bình)

### Hiện trạng
- LR = 1e-3, Adam optimizer
- Early stopping patience = 7
- Batch size = 64
- Gradient clipping = 5.0
- Không dùng learning rate scheduler

### Cải tiến khả thi

| Cải tiến | Dự đoán tác động | Độ khó |
|:---|:---:|:---:|
| **Cosine Annealing LR scheduler** | +1~3% F1 | ⭐ Rất dễ |
| **Warmup + Linear decay** | +1~2% F1 | ⭐ Rất dễ |
| **Label Smoothing** | +1~2% F1 | ⭐⭐ Dễ |
| **Adversarial Training (FGM/PGD)** | +2~4% F1 | ⭐⭐⭐ Trung bình |
| **Multi-task Learning** (aspect + sentiment joint) | +3~5% F1 | ⭐⭐⭐⭐ Khó |
| **Knowledge Distillation** từ PhoBERT → BiGRU | +3~6% F1 | ⭐⭐⭐⭐ Khó |

---

## 🟣 Nhóm 5: Cải Tiến Input/Preprocessing (Tác động: ⭐⭐ Trung bình)

### Hiện trạng
- Tokenization: simple `text.lower().split()` (whitespace split)
- Không word segmentation cho tiếng Việt (VnCoreNLP / underthesea)
- MAX_LEN = 128

### Cải tiến khả thi

| Cải tiến | Dự đoán tác động | Độ khó |
|:---|:---:|:---:|
| **Vietnamese Word Segmentation** (underthesea/VnCoreNLP) | +3~5% F1 | ⭐⭐ Dễ |
| **POS tagging** as auxiliary feature | +2~3% F1 | ⭐⭐⭐ Trung bình |
| **Subword tokenization** (BPE/SentencePiece) | +2~4% F1 | ⭐⭐⭐ Trung bình |
| **ELMo-style contextualized features** | +3~5% F1 | ⭐⭐⭐⭐ Khó |

> [!WARNING]
> Tokenization bằng `text.lower().split()` là một bottleneck nghiêm trọng cho tiếng Việt. Ví dụ: "điện thoại" bị tách thành ["điện", "thoại"] — mất hoàn toàn nghĩa compound word. Dùng **underthesea** để segment sẽ giữ nguyên "điện_thoại" → embedding chính xác hơn nhiều.

---

## 📈 Tổng Hợp: Roadmap Cải Tiến Đề Xuất

### 🥇 Priority 1 — Quick Wins (1-2 ngày, tác động lớn)
1. **Vietnamese Word Segmentation** (`underthesea`) → fix tokenization
2. **FastText pre-trained Vietnamese** thay Word2Vec tự train
3. **Class-weighted CRF loss** → fix data imbalance
4. **LR Scheduler** (Cosine Annealing)

### 🥈 Priority 2 — Moderate Effort (3-5 ngày)
5. **Self-Attention layer** sau BiGRU
6. **Data Augmentation** cho nhãn hiếm
7. **Oversampling** câu chứa nhãn minority
8. **Adversarial Training (FGM)**

### 🥉 Priority 3 — Major Rework (1-2 tuần)
9. **Hybrid CNN-BiGRU-Attention-CRF** pipeline
10. **Multi-task Learning** framework
11. **Knowledge Distillation** từ PhoBERT

---

## 🎯 Ước Tính Tiềm Năng Cải Tiến

| Metric | Hiện tại (BiGRU-CRF) | Sau Priority 1 | Sau Priority 1+2 | Ceiling (lý thuyết) |
|:---|:---:|:---:|:---:|:---:|
| **Token Accuracy** | 69.23% | ~73-75% | ~76-79% | ~85% |
| **Micro F1** | 79.66% | ~83-85% | ~86-88% | ~92% |
| **Macro F1** | 59.18% | ~65-68% | ~70-75% | ~82% |
| **Aspect F1 (Micro)** | 53.97% | ~58-62% | ~65-70% | ~80% |
| **Aspect-Polarity F1 (Micro)** | 51.15% | ~56-60% | ~62-67% | ~78% |

> [!CAUTION]
> Các con số ước tính trên dựa trên kinh nghiệm từ các bài toán ABSA tương đương. **Macro F1** có tiềm năng cải thiện nhiều nhất (~+15-20 điểm) vì hiện tại bị kéo thấp bởi data imbalance — vấn đề hoàn toàn có thể khắc phục được.

---

## 💡 Kết Luận

**Các baseline models hiện tại còn RẤT NHIỀU không gian cải tiến**, chủ yếu từ 3 nguồn:

1. **🔴 Data Imbalance** (tác động lớn nhất, ~+10-15% Macro F1): Mô hình gần như không học được nhãn hiếm. Chỉ cần class-weighted loss + oversampling là đã có thể tăng đáng kể.

2. **🟡 Embedding chất lượng thấp** (~+5-8% F1): Word2Vec 150d train trên 11K câu là quá ít. Thay bằng FastText Vietnamese pre-trained hoặc PhoBERT embeddings sẽ tạo bước nhảy lớn.

3. **🟠 Tokenization thô sơ** (~+3-5% F1): Whitespace split phá vỡ compound words tiếng Việt. Vietnamese word segmentation là must-have.

**Tổng tiềm năng cải tiến ước tính: +10~20% Micro F1, +15~25% Macro F1** trước khi cần chuyển sang Transformer-based models (PhoBERT, Qwen).
