/**
 * Vietnamese ABSA — Frontend Application Logic
 * Handles API calls, result rendering, and UI interactions.
 */

// === Constants ===
const API_BASE = window.location.origin;
const ASPECT_ICONS = {
    "CAMERA": "📷", "FEATURES": "⚡", "PERFORMANCE": "🚀",
    "DESIGN": "🎨", "PRICE": "💰", "GENERAL": "📱",
    "SCREEN": "🖥️", "BATTERY": "🔋", "STORAGE": "💾",
    "SER&ACC": "🛎️",
};
const SENTIMENT_EMOJI = { "POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞" };
const SENTIMENT_VI = { "POSITIVE": "Tích cực", "NEUTRAL": "Trung lập", "NEGATIVE": "Tiêu cực" };

// === DOM Elements ===
const reviewInput = document.getElementById('review-input');
const charCount = document.getElementById('char-count');
const btnAnalyze = document.getElementById('btn-analyze');
const modelSelect = document.getElementById('model-select');
const resultsSection = document.getElementById('results-section');
const emptyState = document.getElementById('empty-state');
const exampleChips = document.getElementById('example-chips');

// Metrics
const metricAspects = document.getElementById('metric-aspects');
const metricSpans = document.getElementById('metric-spans');
const metricOverall = document.getElementById('metric-overall');
const metricOverallIcon = document.getElementById('metric-overall-icon');

// Results
const highlightText = document.getElementById('highlight-text');
const spansList = document.getElementById('spans-list');

// === Event Listeners ===
reviewInput.addEventListener('input', () => {
    charCount.textContent = `${reviewInput.value.length} / 2000`;
});

btnAnalyze.addEventListener('click', handleAnalyze);

// Enter key shortcut (Ctrl/Cmd + Enter)
reviewInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        handleAnalyze();
    }
});

// Example chips
exampleChips.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (chip) {
        reviewInput.value = chip.dataset.text;
        charCount.textContent = `${reviewInput.value.length} / 2000`;
        reviewInput.focus();
    }
});

// === Main Analysis Handler ===
async function handleAnalyze() {
    const text = reviewInput.value.trim();
    if (!text) {
        showToast('⚠️ Vui lòng nhập đánh giá sản phẩm');
        return;
    }

    const model = modelSelect.value;
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, model }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (error) {
        console.error('Prediction error:', error);
        showToast(`❌ ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// === Render Results ===
function renderResults(data) {
    const { spans, summary } = data;

    // Show results, hide empty state
    resultsSection.style.display = 'block';
    emptyState.style.display = 'none';

    // Metrics
    metricAspects.textContent = summary.unique_aspects || 0;
    metricSpans.textContent = summary.total_spans || 0;

    const sc = summary.sentiment_counts || {};
    const pos = sc.POSITIVE || 0;
    const neg = sc.NEGATIVE || 0;
    if (spans.length === 0) {
        metricOverall.textContent = '—';
        metricOverallIcon.textContent = '🤷';
    } else if (pos > neg) {
        metricOverall.textContent = 'Tích cực';
        metricOverallIcon.textContent = '😊';
    } else if (neg > pos) {
        metricOverall.textContent = 'Tiêu cực';
        metricOverallIcon.textContent = '😞';
    } else {
        metricOverall.textContent = 'Trung lập';
        metricOverallIcon.textContent = '😐';
    }

    // Highlighted text
    highlightText.innerHTML = buildHighlightedHTML(data.text, spans);

    // Span cards
    renderSpanCards(spans);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// === Build Highlighted Text HTML ===
function buildHighlightedHTML(text, spans) {
    if (!spans.length) return escapeHtml(text);

    // Sort by start position, resolve overlaps (keep first)
    const sorted = [...spans].sort((a, b) => a.start - b.start);
    const clean = [];
    let lastEnd = -1;
    for (const span of sorted) {
        if (span.start >= lastEnd) {
            clean.push(span);
            lastEnd = span.end;
        }
    }

    let html = '';
    let pos = 0;
    for (const span of clean) {
        if (span.start > pos) {
            html += escapeHtml(text.slice(pos, span.start));
        }
        const hlClass = `hl-${span.sentiment.toLowerCase()}`;
        const icon = ASPECT_ICONS[span.aspect] || '📌';
        html += `<span class="${hlClass}">${escapeHtml(text.slice(span.start, span.end))}<span class="hl-label">${icon}${span.aspect}</span></span>`;
        pos = span.end;
    }
    if (pos < text.length) {
        html += escapeHtml(text.slice(pos));
    }

    return html;
}

// === Render Span Cards ===
function renderSpanCards(spans) {
    if (!spans.length) {
        spansList.innerHTML = '<div class="no-results">❌ Không phát hiện khía cạnh nào</div>';
        return;
    }

    spansList.innerHTML = spans.map((span, i) => {
        const icon = ASPECT_ICONS[span.aspect] || '📌';
        const emoji = SENTIMENT_EMOJI[span.sentiment] || '';
        const sentVi = SENTIMENT_VI[span.sentiment] || span.sentiment;
        const cardClass = `span-card-${span.sentiment.toLowerCase()}`;
        const pillClass = `pill-${span.sentiment.toLowerCase()}`;

        return `
            <div class="span-card ${cardClass}" style="animation-delay: ${i * 0.08}s">
                <div class="span-header">
                    <span class="span-aspect">${icon} ${span.aspect}</span>
                    <span class="span-sentiment ${pillClass}">${emoji} ${sentVi}</span>
                </div>
                <div class="span-text-excerpt">"${escapeHtml(span.text)}"</div>
            </div>
        `;
    }).join('');
}

// === UI Helpers ===
function setLoading(loading) {
    if (loading) {
        btnAnalyze.classList.add('loading');
        btnAnalyze.disabled = true;
    } else {
        btnAnalyze.classList.remove('loading');
        btnAnalyze.disabled = false;
    }
}

function showToast(message) {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => toast.remove(), 4000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// === Init: Check API Health ===
(async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (res.ok) {
            const data = await res.json();
            console.log('✅ API connected:', data);

            // Enable loaded models in selector
            if (data.models_loaded.includes('phobert_crf')) {
                const opt = modelSelect.querySelector('option[value="phobert_crf"]');
                if (opt) opt.disabled = false;
            }
        }
    } catch {
        console.warn('⚠️ API not reachable — make sure the backend is running');
    }
})();
