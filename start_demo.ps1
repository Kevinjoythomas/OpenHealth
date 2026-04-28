# start_demo.ps1 -- one-command demo startup for OpenHealth
# Starts: ngrok tunnel + all backend services
# Frontend is already live on Vercel (static)
#
# Usage: .\start_demo.ps1

$Root   = $PSScriptRoot
$Env    = "$Root\.env"
$Python = "C:\Users\kevin\anaconda3\envs\chromadb\python.exe"
$Ngrok  = "ngrok"

# Load .env into session
Get-Content $Env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}

# Helper: start a service in its own terminal window
function Start-Svc($name, $port) {
    $svcPath = "$Root\services\$name"
    $envBlock = "Get-Content '$Env' | ForEach-Object { if (`$_ -match '^\s*([^#][^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable(`$matches[1].Trim(), `$matches[2].Trim(), 'Process') } }"
    $cmd = "$envBlock; `$env:SERVICE_PORT='$port'; & '$Python' -m app.main"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$svcPath'; $cmd" -WindowStyle Normal
    Write-Host "  Started $name on :$port" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  OpenHealth -- Demo Startup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Ngrok tunnel
Write-Host "Starting ngrok tunnel..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$Ngrok' start backend" -WindowStyle Normal
Write-Host "  ngrok -> https://nonsignificantly-untippled-mikaela.ngrok-free.dev" -ForegroundColor Green
Start-Sleep -Seconds 3

# 2. Backend services
Write-Host ""
Write-Host "Starting backend services..." -ForegroundColor Yellow
Start-Svc "auth-service"      5001; Start-Sleep -Seconds 2
Start-Svc "retrieval-service" 5003; Start-Sleep -Seconds 2
Start-Svc "chat-orchestrator" 5002; Start-Sleep -Seconds 2
Start-Svc "api-gateway"       5000; Start-Sleep -Seconds 3

# 3. Health checks
Write-Host ""
Write-Host "Checking services..." -ForegroundColor Yellow

$services = @(
    @{ name = "API Gateway";  url = "http://localhost:5000/health" },
    @{ name = "Auth";         url = "http://localhost:5001/health" },
    @{ name = "Chat";         url = "http://localhost:5002/health" },
    @{ name = "Retrieval";    url = "http://localhost:5003/health" }
)

foreach ($svc in $services) {
    try {
        Invoke-WebRequest -Uri $svc.url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop | Out-Null
        Write-Host "  OK  $($svc.name)" -ForegroundColor Green
    } catch {
        Write-Host "  !!  $($svc.name) not responding yet (may still be starting)" -ForegroundColor Red
    }
}

# 4. Ollama check
Write-Host ""
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "  OK  Ollama" -ForegroundColor Green
} catch {
    Write-Host "  !!  Ollama not running - start it from the system tray first" -ForegroundColor Red
}

# 5. Summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Demo is live:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Vercel (public) : https://open-health-front.vercel.app" -ForegroundColor White
Write-Host "  Login           : /login" -ForegroundColor White
Write-Host "  Chatbot         : /chat" -ForegroundColor White
Write-Host ""
Write-Host "  Backend (ngrok) : https://nonsignificantly-untippled-mikaela.ngrok-free.dev" -ForegroundColor White
Write-Host ""
Write-Host "  To stop: close the 5 terminal windows that opened" -ForegroundColor DarkGray
Write-Host "==========================================" -ForegroundColor Cyan
