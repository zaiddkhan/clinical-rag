FROM python:3.9

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix the path to match your file structure
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
