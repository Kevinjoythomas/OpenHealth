# Run all OpenHealth services locally on Windows (no Docker)
# Usage: .\run_local.ps1

$Root = $PSScriptRoot
$Env = "$Root\.env"
$Python = "C:\Users\kevin\anaconda3\envs\chromadb\python.exe"

# Install shared dev deps
Write-Host "-> Installing dev dependencies..." -ForegroundColor Cyan
pip install -r "$Root\requirements-dev.txt" -q

# Load .env into current session
Write-Host "-> Loading .env..." -ForegroundColor Cyan
Get-Content $Env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}

# Helper to start a service in a new window
function Start-Service($name, $port) {
    Write-Host "-> Starting $name on :$port ..." -ForegroundColor Green
    $svcPath = "$Root\services\$name"
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "cd '$svcPath'; Get-Content '$Env' | ForEach-Object { if (`$_ -match '^\s*([^#][^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable(`$matches[1].Trim(), `$matches[2].Trim(), 'Process') } }; `$env:SERVICE_PORT='$port'; & '$Python' -m app.main"
    )
}

Start-Service "auth-service"      5001
Start-Sleep 2
Start-Service "retrieval-service" 5003
Start-Sleep 2
Start-Service "chat-orchestrator" 5002
Start-Sleep 2
Start-Service "api-gateway"       5000
Start-Sleep 2

# Start frontend
Write-Host "-> Starting frontend on :3000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$Root\website'; & '$Python' serve.py"
)

Write-Host ""
Write-Host "All services running:" -ForegroundColor Yellow
Write-Host "  Frontend  -> http://localhost:3000/login" -ForegroundColor Green
Write-Host "  API       -> http://localhost:5000" -ForegroundColor Green
Write-Host "  Auth      -> http://localhost:5001" -ForegroundColor Green
Write-Host "  Chat      -> http://localhost:5002" -ForegroundColor Green
Write-Host "  Retrieval -> http://localhost:5003" -ForegroundColor Green
