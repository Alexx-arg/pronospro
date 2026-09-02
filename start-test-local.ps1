# Script de arranque local (Windows PowerShell)
# 1. Configura variables de entorno
$env:ENV = "development"
$env:PYTHONPATH = "C:\Users\Alexxx\Desktop\Proyecto\backend"
$env:DATABASE_URL = "sqlite+aiosqlite:///./test_local.db"
$env:MODEL_PATH = "C:\Users\Alexxx\Desktop\Proyecto\data\models\final\lightgbm_production.joblib"
$env:NVIDIA_API_KEY = ""  # dejar vacío si no tienes, el endpoint /explain responderá 503 controlado

# 2. Instalar dependencias (una vez)
python -m pip install -r backend/requirements.txt

# 3. Crear DB de prueba y seed (una vez)
python backend/seed_test_db.py

# 4. Arrancar API en el puerto 8000
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000