FROM python:3.12-slim-bookworm

ARG RSIHUB_COMMIT=d6554299c58bf0fe2db04ddda0277f9db67ea680
ARG HARNESSRISK_COMMIT=07671f25b515fb936e0312dfdd3ca60b4bdf2f92

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RSIHUB_SRC=/opt/RSIHub/src \
    HARNESSRISK_ROOT=/opt/HarnessRisk \
    HF_HOME=/models/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && git init /opt/RSIHub \
    && git -C /opt/RSIHub remote add origin https://github.com/simple-agent-lab/RSIHub.git \
    && git -C /opt/RSIHub fetch --depth=1 origin "${RSIHUB_COMMIT}" \
    && git -C /opt/RSIHub checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/RSIHub rev-parse HEAD)" = "${RSIHUB_COMMIT}" \
    && git init /opt/HarnessRisk \
    && git -C /opt/HarnessRisk remote add origin https://github.com/Baiyajing/HarnessRisk.git \
    && git -C /opt/HarnessRisk fetch --depth=1 origin "${HARNESSRISK_COMMIT}" \
    && git -C /opt/HarnessRisk checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/HarnessRisk rev-parse HEAD)" = "${HARNESSRISK_COMMIT}" \
    && rm -rf /opt/RSIHub/.git /opt/HarnessRisk/.git \
    && apt-get purge -y --auto-remove git

RUN python -m pip install --no-cache-dir /opt/RSIHub 'huggingface_hub>=0.30,<2'

COPY adapter.py /opt/rsihub_harnessrisk_adapter.py
COPY download_models.py /opt/download_hsrl_models.py
COPY source-pins.json /opt/rsihub_harnessrisk_source_pins.json
COPY harnessrisk_case_manifest.json /opt/harnessrisk_case_manifest.json
COPY entrypoint.sh /usr/local/bin/rsihub-harnessrisk

RUN chmod 0755 /usr/local/bin/rsihub-harnessrisk \
    && mkdir -p /models /data/HarnessRisk /runs

WORKDIR /runs
ENTRYPOINT ["/usr/local/bin/rsihub-harnessrisk"]
CMD ["evolve", "--help"]
