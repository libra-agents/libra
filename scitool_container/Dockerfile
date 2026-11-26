FROM ghcr.io/all-hands-ai/runtime:0.39-nikolaik

WORKDIR /app

COPY requirements-container.txt /app/
RUN pip install --no-cache-dir -r requirements-container.txt 2>&1 | tail -5 && \
    python -c "import rdkit; print('rdkit OK')" && \
    python -c "import numpy; print('numpy OK')"

# SciToolAgent extra deps (best-effort, chatmof excluded - incompatible with Python 3.12)
RUN pip install --no-cache-dir rxn4chemistry MDAnalysis keras || true
RUN pip install --no-cache-dir moffragmentor || true
RUN pip install --no-cache-dir paper-qa || true
RUN pip install --no-cache-dir generate-chroma || true
RUN python -c "import rxn4chemistry; print('rxn4chemistry OK')" || echo "rxn4chemistry: SKIP"; \
    python -c "import MDAnalysis; print('MDAnalysis OK')" || echo "MDAnalysis: SKIP"; \
    python -c "import keras; print('keras OK')" || echo "keras: SKIP"

COPY tool_server.py /app/

ENV TOOLS_DIR=/app/ToolsAgent
ENV TOOL_SERVER_PORT=8100
ENV PYTHONUNBUFFERED=1

EXPOSE 8100

CMD ["python", "/app/tool_server.py"]
