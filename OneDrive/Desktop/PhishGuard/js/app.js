'use strict';

// ═══════════════════════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════════════════════

const classifier = new PhishingClassifier();

document.addEventListener('DOMContentLoaded', () => {
  UI.init();
  classifier.loadModel('./model.json').then(loaded => {
    UI.updateModeBadge(loaded ? 'rf' : 'heuristic');
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// History (localStorage)
// ═══════════════════════════════════════════════════════════════════════════════

const History = (() => {
  const KEY      = 'phishguard_history';
  const MAX_ITEMS = 15;

  function load() {
    try {
      return JSON.parse(localStorage.getItem(KEY)) || [];
    } catch (_) { return []; }
  }

  function save(items) {
    try { localStorage.setItem(KEY, JSON.stringify(items)); } catch (_) {}
  }

  function push(entry) {
    const items = load();
    // Remove duplicate URLs
    const filtered = items.filter(i => i.url !== entry.url);
    filtered.unshift(entry);
    save(filtered.slice(0, MAX_ITEMS));
  }

  function clear() { save([]); }

  return { load, push, clear };
})();

// ═══════════════════════════════════════════════════════════════════════════════
// Gauge animation
// ═══════════════════════════════════════════════════════════════════════════════

const Gauge = (() => {
  const CIRCUMFERENCE = 2 * Math.PI * 72; // r = 72

  let _currentRaf = null;

  function _colorForProbability(p) {
    if (p >= 0.65) return { stroke: '#f85149', text: '#f85149' };
    if (p >= 0.40) return { stroke: '#d29922', text: '#d29922' };
    return                { stroke: '#3fb950', text: '#3fb950' };
  }

  /**
   * Animate the SVG gauge from its current dashoffset to target.
   * @param {number} probability  0–1
   */
  function animate(probability) {
    const fill    = document.getElementById('gaugeFill');
    const pctText = document.getElementById('gaugePercent');
    if (!fill || !pctText) return;

    const targetFraction = Math.max(0, Math.min(1, probability));
    const targetOffset   = CIRCUMFERENCE * (1 - targetFraction);
    const targetPct      = Math.round(targetFraction * 100);
    const color          = _colorForProbability(targetFraction);

    // Cancel any running animation
    if (_currentRaf) cancelAnimationFrame(_currentRaf);

    // Read current state
    let currentOffset = parseFloat(fill.style.strokeDashoffset || CIRCUMFERENCE);
    let currentPct    = parseInt(pctText.textContent, 10) || 0;

    const startTime  = performance.now();
    const DURATION   = 800; // ms

    function step(now) {
      const elapsed  = now - startTime;
      const progress = Math.min(elapsed / DURATION, 1);
      // Ease-out cubic
      const eased    = 1 - Math.pow(1 - progress, 3);

      const newOffset = currentOffset + (targetOffset - currentOffset) * eased;
      const newPct    = Math.round(currentPct + (targetPct - currentPct) * eased);

      fill.style.strokeDashoffset = newOffset;
      fill.style.stroke           = color.stroke;
      pctText.textContent         = newPct + '%';
      pctText.style.fill          = color.text;

      if (progress < 1) {
        _currentRaf = requestAnimationFrame(step);
      } else {
        fill.style.strokeDashoffset = targetOffset;
        pctText.textContent         = targetPct + '%';
      }
    }

    _currentRaf = requestAnimationFrame(step);
  }

  function reset() {
    const fill    = document.getElementById('gaugeFill');
    const pctText = document.getElementById('gaugePercent');
    if (!fill || !pctText) return;
    if (_currentRaf) cancelAnimationFrame(_currentRaf);
    fill.style.strokeDashoffset = CIRCUMFERENCE;
    fill.style.stroke           = '#3fb950';
    pctText.textContent         = '0%';
    pctText.style.fill          = '#8b92a5';
  }

  return { animate, reset, CIRCUMFERENCE };
})();

// ═══════════════════════════════════════════════════════════════════════════════
// UI
// ═══════════════════════════════════════════════════════════════════════════════

const UI = (() => {

  // ── DOM refs ─────────────────────────────────────────────────────────────
  let urlInput, analyzeBtn, clearBtn, errorMsg,
      resultsSection, verdictBadge, verdictUrl,
      confidenceEl, modeTagEl, featuresGrid,
      historySection, historyList, modeBadgeEl,
      gaugeLabel, analyzeLoader;

  function init() {
    urlInput       = document.getElementById('urlInput');
    analyzeBtn     = document.getElementById('analyzeBtn');
    clearBtn       = document.getElementById('clearBtn');
    errorMsg       = document.getElementById('errorMsg');
    resultsSection = document.getElementById('resultsSection');
    verdictBadge   = document.getElementById('verdictBadge');
    verdictUrl     = document.getElementById('verdictUrl');
    confidenceEl   = document.getElementById('confidenceEl');
    modeTagEl      = document.getElementById('modeTagEl');
    featuresGrid   = document.getElementById('featuresGrid');
    historySection = document.getElementById('historySection');
    historyList    = document.getElementById('historyList');
    modeBadgeEl    = document.getElementById('modeBadge');
    gaugeLabel     = document.getElementById('gaugeLabel');
    analyzeLoader  = document.getElementById('analyzeLoader');

    _bindEvents();
    _renderHistory();
  }

  function _bindEvents() {
    // Analyze on button click
    analyzeBtn.addEventListener('click', () => analyzeURL());

    // Analyze on Enter key
    urlInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') analyzeURL();
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
      urlInput.value = '';
      urlInput.focus();
      _hideError();
      clearBtn.classList.remove('visible');
    });

    // Show/hide clear button
    urlInput.addEventListener('input', () => {
      clearBtn.classList.toggle('visible', urlInput.value.length > 0);
      _hideError();
    });

    // Example URL chips
    document.querySelectorAll('.example-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        urlInput.value = chip.dataset.url;
        clearBtn.classList.add('visible');
        _hideError();
        analyzeURL();
      });
    });

    // Clear history button
    const clearHistBtn = document.getElementById('clearHistoryBtn');
    if (clearHistBtn) {
      clearHistBtn.addEventListener('click', () => {
        History.clear();
        _renderHistory();
      });
    }
  }

  // ── URL validation ────────────────────────────────────────────────────────

  function _validateURL(raw) {
    const url = raw.trim();
    if (!url) return { ok: false, msg: 'Please enter a URL to analyse.' };
    if (url.length < 4) return { ok: false, msg: 'URL is too short to analyse.' };
    if (url.length > 2048) return { ok: false, msg: 'URL exceeds maximum length (2048 chars).' };
    // Must look vaguely URL-like
    const hasProto = /^https?:\/\//i.test(url);
    const hasDot   = url.includes('.');
    if (!hasProto && !hasDot) {
      return { ok: false, msg: 'Enter a full URL (e.g., https://example.com).' };
    }
    return { ok: true, msg: '' };
  }

  // ── Core analysis flow ────────────────────────────────────────────────────

  async function analyzeURL() {
    const rawUrl = urlInput.value.trim();
    const val    = _validateURL(rawUrl);

    if (!val.ok) { _showError(val.msg); return; }
    _hideError();

    // Animate button to loading state
    _setAnalyzing(true);

    // Slight artificial delay so the UI feels deliberate (not instant flash)
    await _delay(380);

    try {
      const extracted = extractFeatures(rawUrl);
      const result    = classifier.predict(extracted);
      _displayResults(rawUrl, extracted, result);
      History.push({
        url:         rawUrl,
        probability: result.probability,
        label:       result.label,
        timestamp:   Date.now(),
      });
      _renderHistory();
    } catch (err) {
      console.error('[PhishGuard] Analysis error:', err);
      _showError('Analysis failed. Please check the URL and try again.');
    } finally {
      _setAnalyzing(false);
    }
  }

  function _setAnalyzing(active) {
    analyzeBtn.disabled = active;
    analyzeBtn.classList.toggle('loading', active);
    if (analyzeLoader) analyzeLoader.style.display = active ? 'inline-block' : 'none';
  }

  // ── Result rendering ──────────────────────────────────────────────────────

  function _displayResults(url, extracted, result) {
    const { probability, label, confidenceLabel, mode } = result;
    const pct = Math.round(probability * 100);

    // Gauge
    Gauge.animate(probability);
    if (gaugeLabel) {
      gaugeLabel.textContent = pct + '%';
    }

    // Verdict badge
    const badgeClass = { phishing: 'danger', suspicious: 'warn', legitimate: 'safe' }[label];
    const badgeIcon  = { phishing: '🚨', suspicious: '⚠️', legitimate: '✅' }[label];
    const badgeText  = { phishing: 'PHISHING', suspicious: 'SUSPICIOUS', legitimate: 'SAFE' }[label];

    verdictBadge.className = `verdict-badge ${badgeClass}`;
    verdictBadge.innerHTML = `${badgeIcon} ${badgeText}`;

    // URL display (truncated)
    const displayUrl = url.length > 70 ? url.slice(0, 67) + '…' : url;
    verdictUrl.textContent  = displayUrl;
    verdictUrl.title        = url;

    // Meta
    if (confidenceEl) confidenceEl.textContent = `Confidence: ${confidenceLabel}`;
    if (modeTagEl)    modeTagEl.textContent     = mode === 'rf' ? 'ML Model' : 'Heuristic';

    // Feature cards
    _renderFeatureCards(extracted.features);

    // Show results section with animation
    resultsSection.removeAttribute('hidden');
    requestAnimationFrame(() => {
      resultsSection.classList.add('visible');
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  }

  function _renderFeatureCards(features) {
    featuresGrid.innerHTML = '';

    FEATURE_DISPLAY.forEach((info, i) => {
      const val    = features[info.key];
      const status = info.status(val);
      const icon   = { safe: '✓', warn: '⚠', danger: '✗' }[status];

      const card  = document.createElement('div');
      card.className = `feature-card status-${status}`;
      card.style.animationDelay = `${i * 40}ms`;
      card.innerHTML = `
        <div class="feature-header">
          <span class="feature-icon">${info.icon}</span>
          <span class="feature-label">${info.label}</span>
          <span class="feature-status-icon ${status}">${icon}</span>
        </div>
        <div class="feature-value">${info.format(val)}</div>
        <div class="feature-explain">${info.explain(val)}</div>
      `;
      featuresGrid.appendChild(card);
    });
  }

  // ── History rendering ─────────────────────────────────────────────────────

  function _renderHistory() {
    const items = History.load();

    if (items.length === 0) {
      historySection.style.display = 'none';
      return;
    }

    historySection.style.display = '';
    historyList.innerHTML = '';

    items.forEach(item => {
      const pct   = Math.round(item.probability * 100);
      const cls   = { phishing: 'danger', suspicious: 'warn', legitimate: 'safe' }[item.label] || 'safe';
      const icon  = { phishing: '🚨', suspicious: '⚠️', legitimate: '✅' }[item.label] || '✅';
      const ago   = _timeAgo(item.timestamp);
      const disp  = item.url.length > 60 ? item.url.slice(0, 57) + '…' : item.url;

      const row = document.createElement('div');
      row.className = 'history-row';
      row.innerHTML = `
        <span class="hist-icon">${icon}</span>
        <span class="hist-url" title="${_escHtml(item.url)}">${_escHtml(disp)}</span>
        <span class="hist-pct badge-${cls}">${pct}%</span>
        <span class="hist-ago">${ago}</span>
        <button class="hist-recheck" title="Re-analyse">↺</button>
      `;

      row.querySelector('.hist-recheck').addEventListener('click', () => {
        urlInput.value = item.url;
        clearBtn.classList.add('visible');
        analyzeURL();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });

      historyList.appendChild(row);
    });
  }

  // ── Mode badge ────────────────────────────────────────────────────────────

  function updateModeBadge(mode) {
    if (!modeBadgeEl) return;
    if (mode === 'rf') {
      modeBadgeEl.textContent  = '🤖 ML Model Active';
      modeBadgeEl.className    = 'mode-badge mode-ml';
    } else {
      modeBadgeEl.textContent  = '⚙️ Heuristic Mode';
      modeBadgeEl.className    = 'mode-badge mode-heuristic';
    }
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  function _showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.add('visible');
    urlInput.setAttribute('aria-invalid', 'true');
  }

  function _hideError() {
    errorMsg.textContent = '';
    errorMsg.classList.remove('visible');
    urlInput.removeAttribute('aria-invalid');
  }

  function _delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function _timeAgo(ts) {
    const diff = Date.now() - ts;
    const mins = Math.floor(diff / 60000);
    if (mins < 1)  return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24)  return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  }

  function _escHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  return { init, updateModeBadge };
})();
