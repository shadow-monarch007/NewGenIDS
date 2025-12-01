# Demo Test Script 5: Automated Response System
# Demonstrates automatic threat response and mitigation

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DEMO 5: Automated Response System" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "🤖 Testing Auto-Response for Different Threat Scenarios:" -ForegroundColor Green
Write-Host ""

# Test Case 1: Critical DDoS
Write-Host "=" * 80 -ForegroundColor Gray
Write-Host "Scenario 1: Critical DDoS Attack" -ForegroundColor Red
Write-Host "=" * 80 -ForegroundColor Gray

python -c @"
from src.auto_response import respond_to_threat
threat = {
    'attack_type': 'DDoS',
    'severity': 'Critical',
    'confidence': 0.95,
    'src_ip': '192.168.1.100',
    'features': {'packet_rate': 10000, 'byte_rate': 950000}
}
print(f'Threat: {threat[\"attack_type\"]} from {threat[\"src_ip\"]}')
print(f'Severity: {threat[\"severity\"]} | Confidence: {threat[\"confidence\"]:.0%}')
print('')
actions = respond_to_threat(threat)
print('🔧 Auto-Response Actions Taken:')
for i, action in enumerate(actions, 1):
    print(f'   {i}. {action}')
"@

Write-Host ""

# Test Case 2: High Brute Force
Write-Host "=" * 80 -ForegroundColor Gray
Write-Host "Scenario 2: Brute Force Attack" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Gray

python -c @"
from src.auto_response import respond_to_threat
threat = {
    'attack_type': 'Brute_Force',
    'severity': 'High',
    'confidence': 0.88,
    'src_ip': '10.0.0.55',
    'features': {'packet_rate': 150, 'dst_port': 22}
}
print(f'Threat: {threat[\"attack_type\"]} from {threat[\"src_ip\"]}')
print(f'Severity: {threat[\"severity\"]} | Confidence: {threat[\"confidence\"]:.0%}')
print('')
actions = respond_to_threat(threat)
print('🔧 Auto-Response Actions Taken:')
for i, action in enumerate(actions, 1):
    print(f'   {i}. {action}')
"@

Write-Host ""

# Test Case 3: Medium Port Scan
Write-Host "=" * 80 -ForegroundColor Gray
Write-Host "Scenario 3: Port Scan (Reconnaissance)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Gray

python -c @"
from src.auto_response import respond_to_threat
threat = {
    'attack_type': 'Port_Scan',
    'severity': 'Medium',
    'confidence': 0.76,
    'src_ip': '172.16.0.99',
    'features': {'packet_rate': 50, 'tcp_flags_syn': 1}
}
print(f'Threat: {threat[\"attack_type\"]} from {threat[\"src_ip\"]}')
print(f'Severity: {threat[\"severity\"]} | Confidence: {threat[\"confidence\"]:.0%}')
print('')
actions = respond_to_threat(threat)
print('🔧 Auto-Response Actions Taken:')
for i, action in enumerate(actions, 1):
    print(f'   {i}. {action}')
"@

Write-Host ""

# Show response log
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "📋 Response Log Summary:" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan

if (Test-Path results/auto_response_log.json) {
    $log = Get-Content results/auto_response_log.json | ConvertFrom-Json
    Write-Host ""
    Write-Host "Total Responses Logged: $($log.Count)" -ForegroundColor White
    
    if ($log.Count -gt 0) {
        Write-Host ""
        Write-Host "Recent Responses:" -ForegroundColor Gray
        $log | Select-Object -Last 3 | ForEach-Object {
            Write-Host "  - $($_.timestamp): $($_.action) (Threat: $($_.threat_type))" -ForegroundColor Gray
        }
    }
} else {
    Write-Host ""
    Write-Host "  No response log found yet. Responses will be logged during live operations." -ForegroundColor Gray
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "✅ Demo 5 Complete!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
