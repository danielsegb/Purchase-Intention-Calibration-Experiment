FROM python:3.13.5-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "python scripts/download_data.py && python src/run_experiment.py data/online_shoppers_intention.csv --output-dir outputs && python scripts/verify_results.py --output-dir outputs --strict"]
