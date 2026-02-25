Write-Host "Starting Third Eye Model Evaluation Dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard will open in your default browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

streamlit run dashboard_new.py
