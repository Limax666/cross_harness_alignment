FROM python@sha256:db20d8506e2f15b6db39b40b6015ca46bae855dec6dd326d462cfe79ae972893
RUN python -m pip install --no-cache-dir \
    PyYAML==6.0.3 numpy==2.3.5 pandas==2.3.3 openpyxl==3.1.5 \
    requests==2.32.5 beautifulsoup4==4.14.2 lxml==6.0.2 \
    cryptography==46.0.3 Jinja2==3.1.6 pytest==9.0.1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
