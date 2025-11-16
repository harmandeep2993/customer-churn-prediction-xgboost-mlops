# ===== 1. Base image =====
FROM python:3.12-slim

# ===== 2. Working directory =====
WORKDIR /app

# ===== 3. Copy project =====
COPY . .

# ===== 4. Install dependencies =====
RUN pip install --no-cache-dir -r requirements.txt

# ===== 5. Expose port =====
EXPOSE 8000

# ===== 6. Start FastAPI app =====
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]