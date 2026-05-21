const API_BASE = window.location.origin;
const SENTIMENTS = ["POSITIVE", "NEUTRAL", "NEGATIVE"];

const SENTIMENT_LABELS = {
    POSITIVE: "Tích cực",
    NEUTRAL: "Trung lập",
    NEGATIVE: "Tiêu cực",
};

const SENTIMENT_CLASSES = {
    POSITIVE: "positive",
    NEUTRAL: "neutral",
    NEGATIVE: "negative",
};

const SENTIMENT_COLORS = {
    POSITIVE: "#16a34a",
    NEUTRAL: "#d97706",
    NEGATIVE: "#dc2626",
};

const ASPECT_LABELS = {
    CAMERA: "Camera",
    FEATURES: "Tính năng",
    PERFORMANCE: "Hiệu năng",
    DESIGN: "Thiết kế",
    PRICE: "Giá",
    GENERAL: "Tổng quan",
    SCREEN: "Màn hình",
    BATTERY: "Pin",
    STORAGE: "Bộ nhớ",
    "SER&ACC": "Dịch vụ & phụ kiện",
};

let currentResult = null;

const dom = {
    apiStatus: document.getElementById("api-status"),
    apiStatusText: document.getElementById("api-status-text"),
    modelSelect: document.getElementById("model-select"),
    reviewInput: document.getElementById("review-input"),
    charCount: document.getElementById("char-count"),
    exampleChips: document.getElementById("example-chips"),
    btnAnalyze: document.getElementById("btn-analyze"),
    statAspects: document.getElementById("stat-aspects"),
    statSpans: document.getElementById("stat-spans"),
    statPositive: document.getElementById("stat-positive"),
    statDominant: document.getElementById("stat-dominant"),
    sentimentTotal: document.getElementById("sentiment-total"),
    aspectTotal: document.getElementById("aspect-total"),
    sentimentChart: document.getElementById("sentiment-chart"),
    aspectChart: document.getElementById("aspect-chart"),
    aspectFilter: document.getElementById("aspect-filter"),
    aspectTableBody: document.getElementById("aspect-table-body"),
    emptyState: document.getElementById("empty-state"),
    resultsList: document.getElementById("results-list"),
    toast: document.getElementById("toast"),
};

dom.reviewInput.addEventListener("input", () => {
    dom.charCount.textContent = `${dom.reviewInput.value.length} / 2000`;
});

dom.reviewInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        analyzeText();
    }
});

dom.btnAnalyze.addEventListener("click", analyzeText);
dom.aspectFilter.addEventListener("change", renderDashboard);

dom.exampleChips.addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    dom.reviewInput.value = chip.dataset.text;
    dom.charCount.textContent = `${dom.reviewInput.value.length} / 2000`;
    dom.reviewInput.focus();
});

async function analyzeText() {
    const text = dom.reviewInput.value.trim();
    if (!text) {
        showToast("Vui lòng nhập nội dung đánh giá.");
        return;
    }

    setLoading(true);
    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                model: dom.modelSelect.value,
            }),
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || "Không thể phân tích văn bản.");
        }

        const data = await response.json();
        currentResult = {
            text: data.text,
            modelUsed: data.model_used,
            spans: data.spans || [],
            summary: data.summary || {},
        };
        dom.aspectFilter.value = "ALL";
        renderDashboard();
        document.querySelector(".analysis-area").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        showToast(error.message);
    } finally {
        setLoading(false);
    }
}

function renderDashboard() {
    const summary = buildSummary(currentResult);
    const selectedAspect = dom.aspectFilter.value;
    const spans = currentResult?.spans || [];
    const filteredSpans = selectedAspect === "ALL"
        ? spans
        : spans.filter((span) => span.aspect === selectedAspect);

    updateStats(summary);
    renderAspectFilter(summary.aspectCounts);
    renderAspectTable(summary.aspectCounts);
    renderResultDetail(currentResult, filteredSpans);
    drawSentimentChart(summary.sentimentCounts);
    drawAspectChart(summary.aspectCounts);
}

function buildSummary(result) {
    const sentimentCounts = { POSITIVE: 0, NEUTRAL: 0, NEGATIVE: 0 };
    const aspectCounts = {};
    const spans = result?.spans || [];

    spans.forEach((span) => {
        const sentiment = normalizeSentiment(span.sentiment);
        const aspect = span.aspect || "GENERAL";

        sentimentCounts[sentiment] = (sentimentCounts[sentiment] || 0) + 1;
        if (!aspectCounts[aspect]) {
            aspectCounts[aspect] = { POSITIVE: 0, NEUTRAL: 0, NEGATIVE: 0, total: 0 };
        }
        aspectCounts[aspect][sentiment] += 1;
        aspectCounts[aspect].total += 1;
    });

    return {
        totalSpans: spans.length,
        uniqueAspects: Object.keys(aspectCounts).length,
        sentimentCounts,
        aspectCounts,
    };
}

function updateStats(summary) {
    dom.statAspects.textContent = summary.uniqueAspects;
    dom.statSpans.textContent = summary.totalSpans;
    dom.statPositive.textContent = summary.sentimentCounts.POSITIVE || 0;
    dom.statDominant.textContent = getDominantSentiment(summary.sentimentCounts);
    dom.sentimentTotal.textContent = `${summary.totalSpans} nhãn`;
    dom.aspectTotal.textContent = `${summary.uniqueAspects} nhóm`;
}

function renderAspectFilter(aspectCounts) {
    const current = dom.aspectFilter.value;
    const options = Object.keys(aspectCounts)
        .sort((a, b) => aspectCounts[b].total - aspectCounts[a].total)
        .map((aspect) => `<option value="${escapeHtml(aspect)}">${escapeHtml(getAspectLabel(aspect))}</option>`)
        .join("");

    dom.aspectFilter.innerHTML = `<option value="ALL">Tất cả khía cạnh</option>${options}`;
    dom.aspectFilter.value = Object.prototype.hasOwnProperty.call(aspectCounts, current) ? current : "ALL";
}

function renderAspectTable(aspectCounts) {
    const rows = Object.entries(aspectCounts)
        .sort((a, b) => b[1].total - a[1].total)
        .map(([aspect, counts]) => `
            <tr>
                <td><strong>${escapeHtml(getAspectLabel(aspect))}</strong><span>${escapeHtml(aspect)}</span></td>
                <td class="positive-text">${counts.POSITIVE || 0}</td>
                <td class="neutral-text">${counts.NEUTRAL || 0}</td>
                <td class="negative-text">${counts.NEGATIVE || 0}</td>
                <td>${counts.total}</td>
            </tr>
        `)
        .join("");

    dom.aspectTableBody.innerHTML = rows || `<tr><td colspan="5" class="table-empty">Chưa có khía cạnh được phát hiện</td></tr>`;
}

function renderResultDetail(result, spans) {
    if (!result) {
        dom.emptyState.hidden = false;
        dom.resultsList.innerHTML = "";
        return;
    }

    dom.emptyState.hidden = true;
    const badges = spans.map((span) => {
        const sentiment = normalizeSentiment(span.sentiment);
        return `
            <span class="span-badge ${SENTIMENT_CLASSES[sentiment]}">
                ${escapeHtml(getAspectLabel(span.aspect))}
                <small>${escapeHtml(SENTIMENT_LABELS[sentiment])}</small>
            </span>
        `;
    }).join("");

    dom.resultsList.innerHTML = `
        <article class="result-card">
            <div class="result-head">
                <strong>Kết quả từ ${escapeHtml(result.modelUsed || "model")}</strong>
                <span>${spans.length} đoạn</span>
            </div>
            <p class="highlighted-text">${buildHighlightedHTML(result.text, spans)}</p>
            <div class="badge-row">${badges || '<span class="muted">Không phát hiện khía cạnh phù hợp</span>'}</div>
        </article>
    `;
}

function buildHighlightedHTML(text, spans) {
    if (!spans.length) return escapeHtml(text);

    const sorted = [...spans].sort((a, b) => a.start - b.start);
    const clean = [];
    let lastEnd = -1;
    sorted.forEach((span) => {
        if (Number.isInteger(span.start) && Number.isInteger(span.end) && span.start >= lastEnd) {
            clean.push(span);
            lastEnd = span.end;
        }
    });

    let html = "";
    let pos = 0;
    clean.forEach((span) => {
        const sentiment = normalizeSentiment(span.sentiment);
        if (span.start > pos) html += escapeHtml(text.slice(pos, span.start));
        html += `<mark class="${SENTIMENT_CLASSES[sentiment]}">${escapeHtml(text.slice(span.start, span.end))}<span>${escapeHtml(getAspectLabel(span.aspect))}</span></mark>`;
        pos = span.end;
    });
    if (pos < text.length) html += escapeHtml(text.slice(pos));
    return html;
}

function drawSentimentChart(counts) {
    const canvas = dom.sentimentChart;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const total = SENTIMENTS.reduce((sum, key) => sum + (counts[key] || 0), 0);
    clearCanvas(ctx, width, height);

    if (!total) {
        drawEmptyCanvas(ctx, width, height, "Chưa có dữ liệu");
        return;
    }

    const cx = 128;
    const cy = 128;
    const radius = 82;
    const inner = 48;
    let start = -Math.PI / 2;

    SENTIMENTS.forEach((key) => {
        const value = counts[key] || 0;
        const angle = (value / total) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, radius, start, start + angle);
        ctx.closePath();
        ctx.fillStyle = SENTIMENT_COLORS[key];
        ctx.fill();
        start += angle;
    });

    ctx.beginPath();
    ctx.arc(cx, cy, inner, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.fillStyle = "#111827";
    ctx.font = "700 24px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(total, cx, cy + 8);

    SENTIMENTS.forEach((key, index) => {
        const y = 72 + index * 44;
        ctx.fillStyle = SENTIMENT_COLORS[key];
        ctx.fillRect(250, y - 12, 14, 14);
        ctx.fillStyle = "#374151";
        ctx.font = "600 14px system-ui";
        ctx.textAlign = "left";
        ctx.fillText(SENTIMENT_LABELS[key], 274, y);
        ctx.fillStyle = "#6b7280";
        ctx.font = "500 13px system-ui";
        ctx.fillText(`${counts[key] || 0} (${Math.round(((counts[key] || 0) / total) * 100)}%)`, 274, y + 20);
    });
}

function drawAspectChart(aspectCounts) {
    const canvas = dom.aspectChart;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    clearCanvas(ctx, width, height);

    const entries = Object.entries(aspectCounts)
        .sort((a, b) => b[1].total - a[1].total)
        .slice(0, 7);

    if (!entries.length) {
        drawEmptyCanvas(ctx, width, height, "Chưa có dữ liệu");
        return;
    }

    const max = Math.max(...entries.map(([, counts]) => counts.total), 1);
    const chartLeft = 138;
    const barHeight = 18;
    const gap = 14;
    const chartWidth = width - chartLeft - 44;
    let y = 40;

    ctx.font = "600 13px system-ui";
    entries.forEach(([aspect, counts]) => {
        const total = counts.total;
        const barWidth = (total / max) * chartWidth;
        ctx.fillStyle = "#374151";
        ctx.textAlign = "right";
        ctx.fillText(getAspectLabel(aspect), chartLeft - 14, y + 14);

        let x = chartLeft;
        SENTIMENTS.forEach((sentiment) => {
            const value = counts[sentiment] || 0;
            if (!value) return;
            const segmentWidth = (value / total) * barWidth;
            ctx.fillStyle = SENTIMENT_COLORS[sentiment];
            roundRect(ctx, x, y, segmentWidth, barHeight, 4);
            ctx.fill();
            x += segmentWidth;
        });

        ctx.fillStyle = "#6b7280";
        ctx.textAlign = "left";
        ctx.fillText(String(total), chartLeft + barWidth + 8, y + 14);
        y += barHeight + gap;
    });
}

function clearCanvas(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
}

function drawEmptyCanvas(ctx, width, height, text) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#9ca3af";
    ctx.font = "600 15px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(text, width / 2, height / 2);
}

function roundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
}

function setLoading(isLoading) {
    dom.btnAnalyze.classList.toggle("loading", isLoading);
    dom.btnAnalyze.disabled = isLoading;
    dom.modelSelect.disabled = isLoading;
}

function getDominantSentiment(counts) {
    const total = SENTIMENTS.reduce((sum, key) => sum + (counts[key] || 0), 0);
    if (!total) return "-";
    const top = [...SENTIMENTS].sort((a, b) => (counts[b] || 0) - (counts[a] || 0))[0];
    return SENTIMENT_LABELS[top];
}

function normalizeSentiment(sentiment) {
    return SENTIMENTS.includes(sentiment) ? sentiment : "NEUTRAL";
}

function getAspectLabel(aspect) {
    return ASPECT_LABELS[aspect] || aspect || "Khía cạnh";
}

function showToast(message) {
    dom.toast.textContent = message;
    dom.toast.hidden = false;
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => {
        dom.toast.hidden = true;
    }, 4200);
}

function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value ?? "");
    return div.innerHTML;
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (!response.ok) throw new Error("API unavailable");
        const data = await response.json();
        dom.apiStatus.classList.add("online");
        dom.apiStatusText.textContent = `API sẵn sàng (${data.device || "cpu"})`;

        if (Array.isArray(data.models_loaded)) {
            const phobertOption = dom.modelSelect.querySelector('option[value="phobert_crf"]');
            const bigruOption = dom.modelSelect.querySelector('option[value="bigru_crf"]');
            const hasPhoBert = data.models_loaded.includes("phobert_crf");
            const hasBiGru = data.models_loaded.includes("bigru_crf");

            if (phobertOption) phobertOption.disabled = !hasPhoBert;
            if (bigruOption) bigruOption.disabled = !hasBiGru;
            dom.modelSelect.value = hasPhoBert ? "phobert_crf" : "bigru_crf";
        }
    } catch {
        dom.apiStatus.classList.add("offline");
        dom.apiStatusText.textContent = "API chưa kết nối";
    }
}

renderDashboard();
checkHealth();
