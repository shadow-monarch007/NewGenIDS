"""
Phishing Detector Module
------------------------
Lightweight heuristic + scoring engine for phishing URL/email detection.
No external dependencies; pure Python + regex so it can run inside the IDS
pipeline or be called via an API endpoint in the unified dashboard.

Primary focus: fast triage, not full ML classification. Designed so an ML
model can replace `score_indicators()` later while keeping the same interface.

Public API:
    classify_url(url: str) -> dict
    classify_email(text: str) -> dict  (expects raw email content / headers)

Returned dict keys (common):
    input            : original string
    indicators       : list[str] describing triggered heuristics
    score            : 0-100 numeric risk score
    severity         : LOW|MEDIUM|HIGH|CRITICAL
    is_phishing      : bool (score >= threshold)
    category         : URL|EMAIL
    recommended_steps: list[str] remediation / validation actions

Scoring (heuristic aggregation): each indicator adds weighted points.
Thresholds:
    <25  -> LOW
    25-49 -> MEDIUM
    50-74 -> HIGH
    >=75 -> CRITICAL

Future extension hooks:
    - integrate trained ML model (probability mapping to score)
    - add language model based semantic intent analysis
"""
from __future__ import annotations
import re
import idna
from typing import List, Dict

# Suspicious keywords often seen in phishing lures
PHISH_KEYWORDS = [
    'verify', 'update', 'confirm', 'login', 'secure', 'account', 'password',
    'urgent', 'suspend', 'unlock', 'invoice', 'payment', 'reset', 'alert',
    'important', 'bank', 'limited', 'paypal', 'gift', 'winner', 'bonus'
]

# Brand names commonly spoofed (simplified list)
SPOOFED_BRANDS = [
    'microsoft', 'google', 'facebook', 'apple', 'amazon', 'netflix', 'bank', 'paypal'
]

# Top level domains frequently abused in phishing (example set)
SUSPICIOUS_TLDS = [
    'top', 'xyz', 'click', 'monster', 'gq', 'ml', 'tk', 'cn', 'ru'
]

IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
HEX_PATTERN = re.compile(r'%[0-9a-fA-F]{2}')
UNICODE_CONFUSABLE_PATTERN = re.compile(r'[\u0400-\u04FF]|[\u0500-\u052F]')  # Cyrillic blocks (homoglyph risk)
EMAIL_HEADER_PATTERN = re.compile(r'^(from|subject|reply-to|return-path):', re.IGNORECASE | re.MULTILINE)

# Weight map for scoring indicators
WEIGHTS = {
    'many_subdomains': 15,
    'suspicious_tld': 20,
    'ip_in_domain': 25,
    'contains_brand_plus_extra': 25,
    'keyword_presence': 10,
    'long_url': 10,
    'encoded_hex': 10,
    'unicode_confusables': 30,
    'no_https': 15,
    'at_symbol_in_path': 20,
    'multiple_query_params': 10,
    'dot_login': 25,
    'raw_email_links': 20,
    'spoofed_display_name': 25,
}

DEFENSE_STEPS = [
    'Do not click links; hover to inspect real destination.',
    'Verify sender address matches known official domain.',
    'Contact organization via official channel (phone/portal).',
    'Run URL through threat intelligence / sandbox.',
    'Report to security / SOC for further analysis.',
    'Reset credentials if any were entered on the site.'
]

def _severity(score: int) -> str:
    if score >= 75:
        return 'CRITICAL'
    if score >= 50:
        return 'HIGH'
    if score >= 25:
        return 'MEDIUM'
    return 'LOW'

def _final(score: int, indicators: List[str], category: str, original: str) -> Dict[str, any]:
    return {
        'input': original,
        'category': category,
        'indicators': indicators,
        'score': score,
        'severity': _severity(score),
        'is_phishing': score >= 50,
        'recommended_steps': DEFENSE_STEPS[:4] if score < 75 else DEFENSE_STEPS
    }

def classify_url(url: str) -> Dict[str, any]:
    indicators: List[str] = []
    score = 0

    u = url.strip()
    lower = u.lower()

    # Basic parse
    # Extract domain part between protocol and first slash
    domain_match = re.search(r'^(?:https?://)?([^/]+)', lower)
    domain = domain_match.group(1) if domain_match else lower

    # Strip credentials portion
    if '@' in domain:
        indicators.append('at_symbol_in_path')
        score += WEIGHTS['at_symbol_in_path']
        domain = domain.split('@')[-1]

    # Check HTTPS usage
    if lower.startswith('http://') and not lower.startswith('https://'):
        indicators.append('no_https')
        score += WEIGHTS['no_https']

    # Punycode decoding attempt (IDNA)
    try:
        decoded_labels = []
        for label in domain.split('.'):
            if label.startswith('xn--'):
                decoded_labels.append(idna.decode(label))
            else:
                decoded_labels.append(label)
        decoded_domain = '.'.join(decoded_labels)
    except Exception:
        decoded_domain = domain

    # Subdomain count
    if decoded_domain.count('.') >= 4:
        indicators.append('many_subdomains')
        score += WEIGHTS['many_subdomains']

    # Suspicious TLD
    tld = decoded_domain.split('.')[-1]
    if tld in SUSPICIOUS_TLDS:
        indicators.append('suspicious_tld')
        score += WEIGHTS['suspicious_tld']

    # IP literal domain
    if IP_PATTERN.search(domain):
        indicators.append('ip_in_domain')
        score += WEIGHTS['ip_in_domain']

    # Brand + extra tokens (e.g., paypal-secure-login.com)
    for brand in SPOOFED_BRANDS:
        if brand in decoded_domain and not decoded_domain.endswith(brand + '.com'):
            indicators.append('contains_brand_plus_extra')
            score += WEIGHTS['contains_brand_plus_extra']
            break

    # Phishing keywords in path/query
    for kw in PHISH_KEYWORDS:
        if kw in lower:
            indicators.append('keyword_presence')
            score += WEIGHTS['keyword_presence']
            break

    # Long URL
    if len(u) > 120:
        indicators.append('long_url')
        score += WEIGHTS['long_url']

    # Encoded hex patterns
    if HEX_PATTERN.search(u):
        indicators.append('encoded_hex')
        score += WEIGHTS['encoded_hex']

    # Unicode confusables
    if UNICODE_CONFUSABLE_PATTERN.search(decoded_domain):
        indicators.append('unicode_confusables')
        score += WEIGHTS['unicode_confusables']

    # Dot login pattern
    if '.login' in lower or 'login.' in lower:
        indicators.append('dot_login')
        score += WEIGHTS['dot_login']

    # Many query params
    query_params = lower.split('?')[1].split('&') if '?' in lower else []
    if len(query_params) >= 5:
        indicators.append('multiple_query_params')
        score += WEIGHTS['multiple_query_params']

    return _final(score, indicators, 'URL', url)


def classify_email(text: str) -> Dict[str, any]:
    indicators: List[str] = []
    score = 0
    lower = text.lower()

    # Check presence of raw URLs
    urls = re.findall(r'https?://[^\s]+', text)
    if urls:
        indicators.append('raw_email_links')
        score += WEIGHTS['raw_email_links']

    # Spoofed display name pattern: "Microsoft Support <evil@random.xyz>"
    if any(brand in lower for brand in SPOOFED_BRANDS) and re.search(r'<[^>]+@[^>]+>', text):
        indicators.append('spoofed_display_name')
        score += WEIGHTS['spoofed_display_name']

    # Keywords
    if any(kw in lower for kw in PHISH_KEYWORDS):
        indicators.append('keyword_presence')
        score += WEIGHTS['keyword_presence']

    # Unicode confusables
    if UNICODE_CONFUSABLE_PATTERN.search(text):
        indicators.append('unicode_confusables')
        score += WEIGHTS['unicode_confusables']

    # Encoded hex
    if HEX_PATTERN.search(text):
        indicators.append('encoded_hex')
        score += WEIGHTS['encoded_hex']

    return _final(score, indicators, 'EMAIL', text)


__all__ = ['classify_url', 'classify_email']
