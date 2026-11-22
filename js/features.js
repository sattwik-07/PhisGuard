'use strict';

// ─── Constants ────────────────────────────────────────────────────────────────

const SUSPICIOUS_KEYWORDS = [
  'login','signin','verify','update','secure','account','banking',
  'confirm','password','credential','paypal','ebay','amazon','apple',
  'microsoft','support','alert','auth','wallet','crypto','bitcoin',
  'free','prize','winner','urgent','suspended','limited','locked',
  'validate','recovery','reset','webscr','cmd',
];

const URL_SHORTENERS = [
  'bit.ly','tinyurl.com','goo.gl','t.co','ow.ly','is.gd','buff.ly',
  'adf.ly','bit.do','rb.gy','short.link','tiny.cc','cutt.ly',
  'shorturl.at','shorte.st','clck.ru','snip.ly',
];

const SAFE_TLDS = new Set([
  'com','org','net','edu','gov','mil','int','co','io',
  'uk','us','ca','au','de','fr','jp','in','eu',
]);

const IP_REGEX = /^(\d{1,3}\.){3}\d{1,3}(:\d+)?$/;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Shannon entropy of a string.
 * @param {string} str
 * @returns {number}
 */
function shannonEntropy(str) {
  if (!str || str.length === 0) return 0;
  const freq = Object.create(null);
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    freq[ch] = (freq[ch] || 0) + 1;
  }
  const len = str.length;
  let h = 0;
  for (const ch in freq) {
    const p = freq[ch] / len;
    h -= p * Math.log2(p);
  }
  return h;
}

/**
 * Safely parse a URL string. Handles missing protocol.
 * @param {string} raw
 * @returns {URL|null}
 */
function safeParseURL(raw) {
  try {
    return new URL(raw);
  } catch (_) {}
  try {
    return new URL('https://' + raw);
  } catch (_) {}
  return null;
}

// ─── Main feature extraction ──────────────────────────────────────────────────

/**
 * Extract 29 lexical and structural features from a URL string.
 *
 * @param {string} rawUrl
 * @returns {{
 *   features: Object.<string, number>,
 *   featureArray: number[],
 *   names: string[],
 *   meta: { hostname: string, tld: string, protocol: string }
 * }}
 */
function extractFeatures(rawUrl) {
  const url    = String(rawUrl).trim();
  const parsed = safeParseURL(url);

  const hostname = parsed ? parsed.hostname.toLowerCase()           : '';
  const path     = parsed ? parsed.pathname                         : '';
  const query    = parsed ? parsed.search.replace(/^\?/, '')        : '';
  const protocol = parsed ? parsed.protocol.replace(':', '')        : 'http';
  const urlLower = url.toLowerCase();

  const hostParts = hostname ? hostname.split('.') : [];
  const tld       = hostParts.length >= 2 ? hostParts[hostParts.length - 1] : '';

  // ── Length features ─────────────────────────────────────────────────────
  const urlLen    = url.length;
  const domLen    = hostname.length;
  const pathLen   = path.length;
  const queryLen  = query.length;

  // ── Character count features ─────────────────────────────────────────────
  const countChar = (src, ch) => {
    let n = 0;
    for (let i = 0; i < src.length; i++) if (src[i] === ch) n++;
    return n;
  };
  const matchLen = (src, re) => { const m = src.match(re); return m ? m.length : 0; };

  const numDots        = countChar(url, '.');
  const numHyphens     = countChar(url, '-');
  const numUnderscores = countChar(url, '_');
  const numSlashes     = countChar(url, '/');
  const numAt          = countChar(url, '@');
  const numQuestion    = countChar(url, '?');
  const numAmpersand   = countChar(url, '&');
  const numEquals      = countChar(url, '=');
  const numPercent     = countChar(url, '%');
  const numDigits      = matchLen(url, /\d/g);
  const numLetters     = matchLen(url, /[a-zA-Z]/g);
  const numSpecial     = matchLen(url, /[^a-zA-Z0-9.\-/:_?&=%#@]/g);

  // ── Ratio features ───────────────────────────────────────────────────────
  const digitLetterRatio = numDigits / (numLetters + 1e-6);

  // ── Structural / boolean features ────────────────────────────────────────
  const hasIp          = IP_REGEX.test(hostname)                        ? 1 : 0;
  const hasAt          = numAt > 0                                       ? 1 : 0;
  const hasDoubleSlash = url.slice(url.indexOf('//') + 2).includes('//') ? 1 : 0;
  const usesHttps      = protocol === 'https'                           ? 1 : 0;
  const hasPort        = parsed && parsed.port !== ''                   ? 1 : 0;

  // Subdomain count: exclude 'www' from count
  const subdomainCount = Math.max(
    0,
    hostParts.length - 2 - (hostParts[0] === 'www' ? 1 : 0)
  );

  const suspiciousTld  = SAFE_TLDS.has(tld)                            ? 0 : 1;
  const suspiciousKw   = SUSPICIOUS_KEYWORDS.some(kw => urlLower.includes(kw)) ? 1 : 0;

  // ── Entropy features ─────────────────────────────────────────────────────
  const urlEntropy    = shannonEntropy(url);
  const domainEntropy = shannonEntropy(hostname);

  // ── Misc ─────────────────────────────────────────────────────────────────
  const isShortened  = URL_SHORTENERS.some(s => hostname.includes(s)) ? 1 : 0;
  const numParams    = query ? query.split('&').filter(Boolean).length : 0;

  // ── Assemble feature map ─────────────────────────────────────────────────
  const features = {
    url_length:          urlLen,
    domain_length:       domLen,
    path_length:         pathLen,
    query_length:        queryLen,
    num_dots:            numDots,
    num_hyphens:         numHyphens,
    num_underscores:     numUnderscores,
    num_slashes:         numSlashes,
    num_at:              numAt,
    num_question_marks:  numQuestion,
    num_ampersand:       numAmpersand,
    num_equals:          numEquals,
    num_percent:         numPercent,
    num_digits:          numDigits,
    num_letters:         numLetters,
    num_special_chars:   numSpecial,
    digit_letter_ratio:  digitLetterRatio,
    has_ip_address:      hasIp,
    has_at_symbol:       hasAt,
    has_double_slash:    hasDoubleSlash,
    uses_https:          usesHttps,
    has_port:            hasPort,
    subdomain_count:     subdomainCount,
    suspicious_tld:      suspiciousTld,
    suspicious_keyword:  suspiciousKw,
    url_entropy:         urlEntropy,
    domain_entropy:      domainEntropy,
    is_shortened_url:    isShortened,
    num_query_params:    numParams,
  };

  const names        = Object.keys(features);
  const featureArray = Object.values(features);

  return { features, featureArray, names, meta: { hostname, tld, protocol } };
}

// ─── Feature display metadata ─────────────────────────────────────────────────

/**
 * Human-readable info for each feature, used to build the results UI.
 */
const FEATURE_DISPLAY = [
  {
    key: 'uses_https',
    label: 'HTTPS',
    icon: '🔒',
    format: v => v === 1 ? 'Yes' : 'No',
    status: v => v === 1 ? 'safe' : 'warn',
    explain: v => v === 1
      ? 'Connection is encrypted (HTTPS).'
      : 'No HTTPS — traffic could be intercepted.',
  },
  {
    key: 'has_ip_address',
    label: 'IP as Hostname',
    icon: '🌐',
    format: v => v === 1 ? 'Detected' : 'None',
    status: v => v === 1 ? 'danger' : 'safe',
    explain: v => v === 1
      ? 'URL uses a raw IP address — a strong phishing signal.'
      : 'Hostname is a domain name, not an IP.',
  },
  {
    key: 'has_at_symbol',
    label: '@ Symbol',
    icon: '@',
    format: v => v === 1 ? 'Found' : 'None',
    status: v => v === 1 ? 'danger' : 'safe',
    explain: v => v === 1
      ? '"@" in URL causes browsers to ignore everything before it — a classic trick.'
      : 'No "@" symbol in URL.',
  },
  {
    key: 'has_double_slash',
    label: 'Double-Slash Redirect',
    icon: '//',
    format: v => v === 1 ? 'Found' : 'None',
    status: v => v === 1 ? 'danger' : 'safe',
    explain: v => v === 1
      ? 'Double "//" after the domain indicates a redirect attempt.'
      : 'No suspicious double-slash redirect.',
  },
  {
    key: 'suspicious_keyword',
    label: 'Suspicious Keywords',
    icon: '🔑',
    format: v => v === 1 ? 'Found' : 'None',
    status: v => v === 1 ? 'danger' : 'safe',
    explain: v => v === 1
      ? 'URL contains phishing-associated words (login, verify, secure, etc.).'
      : 'No suspicious phishing keywords detected.',
  },
  {
    key: 'url_length',
    label: 'URL Length',
    icon: '📏',
    format: v => `${v} chars`,
    status: v => v > 100 ? 'danger' : v > 75 ? 'warn' : 'safe',
    explain: v => v > 100
      ? `Very long URL (${v} chars) — often used to hide the true destination.`
      : v > 75
        ? `Moderately long URL (${v} chars) — slightly unusual.`
        : `Normal URL length (${v} chars).`,
  },
  {
    key: 'url_entropy',
    label: 'URL Entropy',
    icon: '📊',
    format: v => v.toFixed(2),
    status: v => v > 4.8 ? 'danger' : v > 4.2 ? 'warn' : 'safe',
    explain: v => v > 4.8
      ? `High entropy (${v.toFixed(2)}) — URL appears randomised or obfuscated.`
      : v > 4.2
        ? `Moderate entropy (${v.toFixed(2)}) — slightly complex URL structure.`
        : `Normal entropy (${v.toFixed(2)}) — URL structure looks natural.`,
  },
  {
    key: 'subdomain_count',
    label: 'Subdomain Depth',
    icon: '🏗️',
    format: v => v === 0 ? 'None' : `${v} level${v > 1 ? 's' : ''}`,
    status: v => v > 3 ? 'danger' : v > 1 ? 'warn' : 'safe',
    explain: v => v > 3
      ? `Deep subdomain nesting (${v} levels) — used to mimic legitimate domains.`
      : v > 1
        ? `${v} subdomain levels — worth checking.`
        : 'Normal subdomain structure.',
  },
  {
    key: 'suspicious_tld',
    label: 'TLD Risk',
    icon: '🔖',
    format: v => v === 1 ? 'Unusual' : 'Standard',
    status: v => v === 1 ? 'warn' : 'safe',
    explain: v => v === 1
      ? 'Non-standard TLD — commonly used in phishing (e.g., .tk, .ml, .ga).'
      : 'Standard, well-known TLD.',
  },
  {
    key: 'is_shortened_url',
    label: 'URL Shortener',
    icon: '✂️',
    format: v => v === 1 ? 'Detected' : 'None',
    status: v => v === 1 ? 'warn' : 'safe',
    explain: v => v === 1
      ? 'URL shortening service detected — hides the real destination.'
      : 'No URL shortening service detected.',
  },
  {
    key: 'num_hyphens',
    label: 'Hyphens in URL',
    icon: '—',
    format: v => v,
    status: v => v > 5 ? 'danger' : v > 3 ? 'warn' : 'safe',
    explain: v => v > 5
      ? `${v} hyphens — heavy hyphenation mimics legitimate brand names.`
      : v > 3
        ? `${v} hyphens — slightly elevated.`
        : `${v} hyphen(s) — within normal range.`,
  },
  {
    key: 'has_port',
    label: 'Non-Standard Port',
    icon: '🔌',
    format: v => v === 1 ? 'Detected' : 'None',
    status: v => v === 1 ? 'warn' : 'safe',
    explain: v => v === 1
      ? 'Non-standard port in URL — unusual for legitimate web services.'
      : 'Standard port (no explicit port in URL).',
  },
];
