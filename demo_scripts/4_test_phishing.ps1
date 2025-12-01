# Demo Test Script 4: Phishing Detection
# Tests URL and email phishing detection

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DEMO 4: Phishing Detection" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Test URLs
Write-Host "🔗 Testing URL Phishing Detection:" -ForegroundColor Green
Write-Host "=" * 40 -ForegroundColor Gray

$testUrls = @(
    "https://www.google.com",
    "http://paypal-verify-account.suspicious-site.ru/login",
    "https://github.com/legitimate/repo",
    "http://bit.ly/win-prize-now",
    "http://amazon-security-alert.tk/signin"
)

foreach ($url in $testUrls) {
    Write-Host ""
    Write-Host "Testing: $url" -ForegroundColor White
    
    python -c @"
from src.phishing_detector import classify_url
result = classify_url('$url')
is_phishing = result['is_phishing']
confidence = result['confidence']
risk = result['risk_level']
emoji = '🚨' if is_phishing else '✅'
print(f'{emoji} Result: {"PHISHING" if is_phishing else "LEGITIMATE"}')
print(f'   Confidence: {confidence:.2%}')
print(f'   Risk Level: {risk}')
"@
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "📧 Testing Email Phishing Detection:" -ForegroundColor Green
Write-Host "=" * 40 -ForegroundColor Gray

$testEmails = @(
    "Your meeting has been scheduled for tomorrow at 2 PM. Please confirm.",
    "URGENT: Your account has been suspended! Click here IMMEDIATELY to verify: http://verify-now.com",
    "Thank you for your purchase. Your order will arrive in 3-5 business days.",
    "You've won $1,000,000! Claim your prize by clicking this link and entering your bank details!"
)

$emailNum = 1
foreach ($email in $testEmails) {
    Write-Host ""
    Write-Host "Email #${emailNum}: $(if ($email.Length -gt 60) {$email.Substring(0,60) + '...'} else {$email})" -ForegroundColor White
    
    $escapedEmail = $email -replace '"', '\"' -replace "'", "\'"
    
    python -c @"
from src.phishing_detector import classify_email
result = classify_email('$escapedEmail')
is_phishing = result['is_phishing']
confidence = result['confidence']
risk = result['risk_level']
emoji = '🚨' if is_phishing else '✅'
print(f'{emoji} Result: {"PHISHING" if is_phishing else "LEGITIMATE"}')
print(f'   Confidence: {confidence:.2%}')
print(f'   Risk Level: {risk}')
"@
    
    $emailNum++
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "✅ Demo 4 Complete!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
