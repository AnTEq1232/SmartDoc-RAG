# Używamy lekkiego, oficjalnego obrazu Pythona
FROM python:3.11-slim

# Ustawienie katalogu roboczego w kontenerze
WORKDIR /app

# Zapobieganie buforowaniu wyjścia oraz tworzeniu plików .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalacja narzędzi systemowych potrzebnych do kompilacji niektórych bibliotek C
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopiujemy i instalujemy zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy resztę kodu aplikacji
COPY . .

# Otwieramy domyślny port Streamlita
EXPOSE 8501

# Uruchamiamy aplikację Streamlit
ENV PYTHONPATH=/app
CMD ["python", "-m", "streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]