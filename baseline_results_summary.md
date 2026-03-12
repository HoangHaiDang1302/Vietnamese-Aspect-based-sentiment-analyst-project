# Bảng Tổng Hợp Kết Quả Các Mô Hình Baseline Đánh Giá Khía Cạnh - Cảm Xúc (Word2Vec)

Dưới đây là bảng tóm tắt đối sánh hiệu năng của 6 mô hình Mạng Neural cơ sở kết hợp với lớp giải mã CRF (Training Joint-Extraction). Kết quả được ghi nhận trên tập Test.

## 1. Bảng Xếp Hạng Hiệu Năng Tổng Thể (Final Leaderboard)

| Mô hình       | Token Acc  |  Micro F1  |  Macro F1  | Weighted F1 |
| :------------ | :--------: | :--------: | :--------: | :---------: |
| **BiGRU-CRF** | **0.6923** | **0.7966** | **0.5918** | **0.7925**  |
| BiLSTM-CRF    |   0.6779   |   0.7853   |   0.5850   |   0.7811    |
| TextCNN-CRF   |   0.6772   |   0.7858   |   0.5541   |   0.7776    |
| GRU-CRF       |   0.6500   |   0.7639   |   0.5348   |   0.7539    |
| LSTM-CRF      |   0.6576   |   0.7622   |   0.5160   |   0.7474    |
| RNN-CRF       |   0.6515   |   0.7614   |   0.4897   |   0.7406    |

> 🏆 **Best Baseline Model:** `BiGRU-CRF` (Micro F1: **79.66%**)

## 2. Nhận xét Tóm Tắt Từ Số Liệu Chi Tiết Của BiGRU-CRF:

- Mạng 2 chiều (Bidirectional) như **BiGRU-CRF** và **BiLSTM-CRF** cho kết quả vượt trội hơn rõ ràng so với các mạng 1 chiều (RNN, LSTM, GRU). Khả năng nắm ngữ cảnh từ cả quá khứ lẫn tương lai đặc biệt quan trọng để bắt lấy chính xác cảm xúc.
- **Các nhãn làm tốt (F1 > 85%):** BATTERY#POSITIVE (88.76%), GENERAL#POSITIVE (88.25%), PERFORMANCE#POSITIVE (87.17%), CAMERA#POSITIVE (86.14%).
- **Các nhãn làm rất tệ (Data Imbalance - Cần tập trung khắc phục mạnh):** DESIGN#NEUTRAL (25%), PRICE#NEUTRAL (27.18%), STORAGE (Positive/Neutral/Negative đều thấp do Support rất bé < 15 mẫu dữ liệu).
- **Khả năng học ranh giới (Span Boundary):** TextCNN-CRF là đối thủ đáng gờm vì nó dự đoán chẻ nhỏ khá gắt (tông cộng 7156 spans - nhiều nhất) nhưng khả năng nắm bắt Cảm xúc vĩ mô (Macro F1) bị thọt hẳn (55% so với BiGRU 59%). BiGRU là mô hình có độ cân bằng tuyệt vời nhất giữa việc bóc mảng (Extraction) và phân loại cảm xúc (Classification).
