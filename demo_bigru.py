"""
===============================================================================
🚀 BiGRU-CRF Joint-Extraction Demo
===============================================================================
Streamlit demo for Aspect-Based Sentiment Analysis using the best Baseline Model.

Run: streamlit run demo_bigru.py
===============================================================================
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from gensim.models import Word2Vec
from torchcrf import CRF

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="BiGRU-CRF ABSA Demo",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .hero-header {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 40px rgba(255, 65, 108, 0.3);
    }
    .hero-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .hero-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem; font-weight: 300; }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .metric-card .metric-value { font-size: 1.8rem; font-weight: 700; color: #2d3748; }
    .metric-card .metric-label { font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 1px; }

    .span-card {
        border-radius: 14px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s ease;
    }
    .span-card:hover { transform: translateX(4px); }
    .span-card-positive { background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%); border-left-color: #38a169; }
    .span-card-neutral { background: linear-gradient(135deg, #fffaf0 0%, #fefce8 100%); border-left-color: #d69e2e; }
    .span-card-negative { background: linear-gradient(135deg, #fff5f5 0%, #fef2f2 100%); border-left-color: #e53e3e; }
    .span-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .span-aspect { font-weight: 700; font-size: 1rem; color: #2d3748; display: flex; align-items: center; gap: 8px; }
    .span-sentiment { padding: 4px 14px; border-radius: 50px; font-weight: 600; font-size: 0.8rem; }
    .pill-positive { background: #c6f6d5; color: #22543d; }
    .pill-neutral { background: #fefcbf; color: #744210; }
    .pill-negative { background: #fed7d7; color: #742a2a; }
    .span-text {
        background: rgba(0,0,0,0.04); border-radius: 8px; padding: 8px 12px;
        font-style: italic; color: #4a5568; font-size: 0.92rem; border: 1px dashed rgba(0,0,0,0.1);
    }

    .highlight-container {
        background: black; border-radius: 12px; padding: 1.5rem; line-height: 2;
        font-size: 1.05rem; border: 1px solid #e2e8f0; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    .highlight-positive { background: linear-gradient(135deg, #c6f6d5, #9ae6b4); padding: 2px 6px; border-radius: 4px; font-weight: 500; border-bottom: 2px solid black; }
    .highlight-neutral { background: linear-gradient(135deg, #fefcbf, #faf089); padding: 2px 6px; border-radius: 4px; font-weight: 500; border-bottom: 2px solid black; }
    .highlight-negative { background: linear-gradient(135deg, #fed7d7, #feb2b2); padding: 2px 6px; border-radius: 4px; font-weight: 500; border-bottom: 2px solid black; }
    .highlight-label { font-size: 0.65rem; font-weight: 700; vertical-align: super; margin-left: 2px; opacity: 0.7; color: #4a5568; }

    .stTextArea textarea { border-radius: 12px !important; border: 2px solid #e2e8f0 !important; font-size: 1rem !important; padding: 1rem !important; }
    .stTextArea textarea:focus { border-color: #FF416C !important; box-shadow: 0 0 0 3px rgba(255, 65, 108, 0.2) !important; }
    .stButton > button { background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important; border: none !important; border-radius: 12px !important; padding: 0.7rem 2.5rem !important; font-weight: 600 !important; color: white !important; box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4) !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; }
    .custom-divider { height: 3px; background: linear-gradient(90deg, #FF416C, #FF4B2B, #FF416C); border: none; border-radius: 2px; margin: 1.5rem 0; }
    .legend { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.85rem; color: #4a5568; }
    .legend-dot { width: 14px; height: 14px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PIPELINE CONFIG & MODELS
# ============================================================
DATA_DIR = '.'
SAVE_DIR = './dl_all_models_bio_crf'
MAX_LEN = 128
W2V_DIM = 150
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
PAD_IDX = 0
UNK_IDX = 1

ASPECTS = ["CAMERA","FEATURES","PERFORMANCE","DESIGN","PRICE",
           "GENERAL","SCREEN","BATTERY","STORAGE","SER&ACC"]
SENTIMENTS = ["POSITIVE","NEUTRAL","NEGATIVE"]
LABEL_NAMES = [f"{a}#{s}" for a in ASPECTS for s in SENTIMENTS]

BIO_TAGS = ['O']
for label in LABEL_NAMES:
    BIO_TAGS.append(f'B-{label}')
    BIO_TAGS.append(f'I-{label}')
NUM_TAGS = len(BIO_TAGS)

class SequenceCRF(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained_emb=None, n_layers=2, dropout=0.3, pad_idx=0, rnn_type='lstm', bidir=True):
        super().__init__()
        self.bidir = bidir
        rnn_out = hidden_dim * 2 if bidir else hidden_dim
        
        if pretrained_emb is not None:
            self.emb = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=pad_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0, bidirectional=bidir)
        self.hidden2tag = nn.Sequential(nn.Linear(rnn_out, rnn_out // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(rnn_out // 2, num_tags))
        self.crf = CRF(num_tags, batch_first=True)
    
    def _get_emissions(self, seqs, lens):
        emb = self.drop(self.emb(seqs))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=seqs.size(1))
        return self.hidden2tag(self.drop(output))
    
    def forward(self, seqs, lens, mask, tags=None):
        emissions = self._get_emissions(seqs, lens)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            best_tags = self.crf.decode(emissions, mask=mask)
            return {'tags': best_tags}

def bio_tags_to_spans(tag_ids, max_tokens):
    spans = []
    current_label = None
    current_start = None
    for t in range(min(len(tag_ids), max_tokens)):
        tag_id = tag_ids[t]
        tag_name = BIO_TAGS[tag_id] if tag_id < len(BIO_TAGS) else 'O'
        if tag_name.startswith('B-'):
            if current_label is not None: spans.append((current_label, current_start, t))
            current_label = tag_name[2:]
            current_start = t
        elif tag_name.startswith('I-'):
            label = tag_name[2:]
            if current_label != label:
                if current_label is not None: spans.append((current_label, current_start, t))
                current_label = label
                current_start = t
        else:
            if current_label is not None:
                spans.append((current_label, current_start, t))
                current_label = None
    if current_label is not None:
        spans.append((current_label, current_start, len(tag_ids)))
    return spans

# ============================================================
# CACHED DATA & MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_all():
    texts = []
    for split in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:
        path = os.path.join(DATA_DIR, split)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    texts.append(json.loads(line.strip())['text'])
                    
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    all_sentences = [t.lower().split() for t in texts]
    
    st.write("🔄 Đang chuẩn bị Word2Vec...")
    w2v = Word2Vec(all_sentences, vector_size=W2V_DIM, window=5, min_count=2, workers=4, epochs=20, sg=1, seed=42)
    
    word2idx = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
    for i, w in enumerate(w2v.wv.index_to_key): word2idx[w] = i + 2
    VOCAB_SIZE = len(word2idx)
    
    emb_matrix = np.random.normal(0, 0.1, (VOCAB_SIZE, W2V_DIM)).astype(np.float32)
    emb_matrix[PAD_IDX] = 0
    for w, idx in word2idx.items():
        if w in w2v.wv: emb_matrix[idx] = w2v.wv[w]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SequenceCRF(vocab_size=VOCAB_SIZE, emb_dim=W2V_DIM, hidden_dim=HIDDEN_DIM, 
                        num_tags=NUM_TAGS, pretrained_emb=emb_matrix, 
                        n_layers=NUM_LAYERS, dropout=DROPOUT, pad_idx=PAD_IDX, 
                        rnn_type='gru', bidir=True).to(device)
    
    model_path = os.path.join(SAVE_DIR, 'bigru_crf.pt')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        st.warning(f"⚠️ Không tìm thấy file weights ({model_path}). Kết quả sẽ ra random!")
        
    model.eval()
    
    return model, word2idx, device

# ============================================================
# PREDICTION LOGIC
# ============================================================
def predict_model(text, model, word2idx, device):
    words = text.lower().split()[:MAX_LEN]
    seq = [word2idx.get(w, UNK_IDX) for w in words]
    seq_len = len(seq)
    
    if seq_len == 0: return []
    
    tensor_seq = torch.tensor([seq]).to(device)
    tensor_len = torch.tensor([seq_len])
    tensor_mask = torch.ones(1, seq_len, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        preds = model(tensor_seq, tensor_len, tensor_mask)
        tags = preds['tags'][0][:seq_len]
        spans = bio_tags_to_spans(tags, seq_len)
        
    pos = 0
    text_lower = text.lower()
    char_positions = []
    
    for w in words:
        idx = text_lower.find(w, pos)
        if idx == -1: idx = pos
        char_positions.append((idx, idx + len(w)))
        pos = idx + len(w)
        
    results = []
    for j, (label_str, s, e_exc) in enumerate(spans):
        # label_str is "ASPECT#SENTIMENT"
        if "#" not in label_str: continue
        aspect, sentiment = label_str.split('#')
        
        start_char = char_positions[s][0]
        end_char = char_positions[e_exc - 1][1]
        
        results.append({
            "aspect": aspect,
            "sentiment": sentiment,
            "text": text[start_char:end_char].strip(),
            "start": start_char,
            "end": end_char
        })
        
    return results

# ============================================================
# MAIN UI
# ============================================================
ASPECT_ICONS = {
    "CAMERA": "📷", "FEATURES": "⚡", "PERFORMANCE": "🚀",
    "DESIGN": "🎨", "PRICE": "💰", "GENERAL": "📱",
    "SCREEN": "🖥️", "BATTERY": "🔋", "STORAGE": "💾",
    "SER&ACC": "🛎️",
}
SENTIMENT_EMOJI = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞"}
SENTIMENT_LABEL_VI = {"POSITIVE": "Tích cực", "NEUTRAL": "Trung lập", "NEGATIVE": "Tiêu cực"}
PILL_CLASS = {"POSITIVE": "pill-positive", "NEUTRAL": "pill-neutral", "NEGATIVE": "pill-negative"}
CARD_CLASS = {"POSITIVE": "span-card-positive", "NEUTRAL": "span-card-neutral", "NEGATIVE": "span-card-negative"}
HIGHLIGHT_CLASS = {"POSITIVE": "highlight-positive", "NEUTRAL": "highlight-neutral", "NEGATIVE": "highlight-negative"}

def build_highlighted_text(text, spans):
    if not spans: return text
    sorted_spans = sorted(spans, key=lambda x: x["start"])
    clean_spans = []
    last_end = -1
    for span in sorted_spans:
        if span["start"] >= last_end:
            clean_spans.append(span)
            last_end = span["end"]

    html_parts = []
    last_pos = 0
    for span in clean_spans:
        if span["start"] > last_pos:
            html_parts.append(text[last_pos:span["start"]])
        hl_class = HIGHLIGHT_CLASS.get(span["sentiment"], "")
        icon = ASPECT_ICONS.get(span["aspect"], "📌")
        html_parts.append(
            f'<span class="{hl_class}">'
            f'{text[span["start"]:span["end"]]}'
            f'<span class="highlight-label">{icon}{span["aspect"]}</span>'
            f'</span>'
        )
        last_pos = span["end"]
    if last_pos < len(text):
        html_parts.append(text[last_pos:])
    return "".join(html_parts)

EXAMPLES = [
    "Sp ổn, mỗi tội vân tay lúc nhận lúc không, nhân viên nhiệt tình, pin trâu, cả đêm tụt 1%",
    "Mua cho mẹ sài nên củng không đòi hỏi gì nhiều, máy đẹp camera siêu ảo, thử chiến game củng ok,pin sài dc 2 ngày với luot wep xem fim, nhân viên tgdd an minh kg phục vụ qua nhiệt tình cho 5*",
    "Máy xài tốt, mượt, sạc rất nhanh, pin trâu, mình dùng tác vụ bình thường (zalo, fb, youtube) thì được 1 ngày rưỡi. camera thì đẹp ảo 😁.",
]

def main():
    st.markdown("""
    <div class="hero-header">
        <h1>🇻🇳 BiGRU-CRF ABSA Demo</h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("⏳ Đang khởi tạo Word2Vec và load Models (Chỉ chạy lần đầu)..."):
        model, word2idx, device = load_all()
        
    with st.sidebar:
        st.markdown("### ⚙️ System Info")
        device_name = "🟢 GPU" if device.type == 'cuda' else "🔵 CPU"
        st.markdown(f"**Device:** {device_name}")
        st.markdown("**Kiến trúc:** `BiGRU-CRF` (Joint Model)")
        st.markdown("**Số Tầng/Hidden:** `2 Layers, 256 Dims`")
        st.markdown(f"**Vocab Size:** `{len(word2idx)}`")
        st.markdown("---")
        st.markdown("#### 🏷️ 10 Khía cạnh (Aspects)")
        for asp in ASPECTS:
            icon = ASPECT_ICONS.get(asp, "📌")
            st.write(f"{icon} {asp}")

    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    def set_example(ex_text):
        st.session_state["user_input"] = ex_text

    col_input, col_examples = st.columns([3, 2])
    with col_input:
        user_text = st.text_area(
            "📝 Nhập đánh giá sản phẩm",
            placeholder="Ví dụ: máy đẹp cực kỳ, camera chụp siêu ảo...",
            height=150,
            key="user_input",
            label_visibility="collapsed"
        )
    with col_examples:
        st.markdown("### 💡 Câu ví dụ (Nhấp để chọn)")
        for i, ex in enumerate(EXAMPLES):
            st.button(f"📌 {ex[:50]}...", key=f"ex_{i}", use_container_width=True, on_click=set_example, args=(ex,))

    analyze_clicked = st.button("🔍 Phân tích", use_container_width=True)

    if analyze_clicked and user_text.strip():
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        spans = predict_model(user_text.strip(), model, word2idx, device)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Khía cạnh tìm thấy</div><div class="metric-value">{len(set(s["aspect"] for s in spans))}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Số đoạn phát hiện</div><div class="metric-value">{len(spans)}</div></div>', unsafe_allow_html=True)
        with col3:
            if spans:
                pos = sum(1 for s in spans if s["sentiment"] == "POSITIVE")
                neg = sum(1 for s in spans if s["sentiment"] == "NEGATIVE")
                overall = "😊 Tích cực" if pos > neg else "😞 Tiêu cực" if neg > pos else "😐 Trung lập"
            else:
                overall = "Không rõ"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Tổng quan đánh giá</div><div class="metric-value">{overall}</div></div>', unsafe_allow_html=True)
            
        st.markdown("")
        col_text, col_spans = st.columns([3, 2])

        with col_text:
            st.markdown("### 🎨 Văn bản được Highlight")
            st.markdown("""
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background: #c6f6d5; border-bottom: 2px solid #38a169;"></div> Tích cực</div>
                <div class="legend-item"><div class="legend-dot" style="background: #fefcbf; border-bottom: 2px solid #d69e2e;"></div> Trung lập</div>
                <div class="legend-item"><div class="legend-dot" style="background: #fed7d7; border-bottom: 2px solid #e53e3e;"></div> Tiêu cực</div>
            </div>
            """, unsafe_allow_html=True)

            highlighted = build_highlighted_text(user_text.strip(), spans)
            st.markdown(f'<div class="highlight-container">{highlighted}</div>', unsafe_allow_html=True)

        with col_spans:
            st.markdown("### 🏷️ Chi tiết bóc tách (Extraction)")
            if spans:
                for span in spans:
                    icon = ASPECT_ICONS.get(span["aspect"], "📌")
                    emoji = SENTIMENT_EMOJI.get(span["sentiment"], "")
                    pill = PILL_CLASS.get(span["sentiment"], "")
                    card = CARD_CLASS.get(span["sentiment"], "")
                    sent_vi = SENTIMENT_LABEL_VI.get(span["sentiment"], span["sentiment"])
                    st.markdown(f"""
                    <div class="span-card {card}">
                        <div class="span-header">
                            <span class="span-aspect">{icon} {span["aspect"]}</span>
                            <span class="span-sentiment {pill}">{emoji} {sent_vi}</span>
                        </div>
                        <div class="span-text">"{span["text"]}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("❌ Không phát hiện thuộc tính/khía cạnh nào trong câu văn!")
                
    elif analyze_clicked:
        st.warning("⚠️ Vui lòng nhập văn bản để mô hình tiến hành phân tích.")

if __name__ == "__main__":
    main()
