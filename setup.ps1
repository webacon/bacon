# setup.ps1 - Windows PowerShell Setup

Write-Host "🥓 BACON ECOSYSTEM SETUP" -ForegroundColor Yellow
Write-Host "=========================" -ForegroundColor Yellow

# Get Supabase key from user
$supabaseUrl = "https://aqicrgckmuqavbxxlwl.supabase.co"
$supabaseKey = Read-Host "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFxaWNyZ2NrbXF1cWF2Ynh4bHdsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUyMDYzODUsImV4cCI6MjA4MDc4MjM4NX0.gQwdg-AO4HmjKgHCC42IOr70i6rpP1RVqxcgGUo9owI"

# Create .env for Bacon Station V3
Write-Host "`n📝 Creating bacon-station-v3/.env..." -ForegroundColor Cyan

$envContent = @"
SUPABASE_URL=$supabaseUrl
SUPABASE_ANON_KEY=$supabaseKey
"@

$envContent | Out-File -FilePath ".\bacon-station-v3\.env" -Encoding UTF8 -Force

# Create .env for BaconAlgo Web
Write-Host "📝 Creating baconalgo-web/.env..." -ForegroundColor Cyan
$envContent | Out-File -FilePath ".\baconalgo-web\.env" -Encoding UTF8 -Force

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "`n🚀 To run apps:" -ForegroundColor Yellow
Write-Host "Desktop: cd bacon-station-v3; streamlit run bacon_station_v3_ultimate.py"
Write-Host "Web: cd baconalgo-web; streamlit run app.py --server.port 8502"
