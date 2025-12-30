FROM docker.cnb.cool/cnb/cool/default-dev-env/dockerfile-caches:ba8c7bf15bfb07dec83820d9a3878fa3134a09a3

# 保险起见：确认 python3/pip 存在（大多数 dev-env 已经有）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev bash git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# venv环境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 装依赖：你后续想切 http client 或换 UI 都方便
RUN python3 -m pip install -U pip \
    && python3 -m pip install "openai>=1.0.0" "textual>=0.70.0" "httpx>=0.25.0" "markitdown[all]" "playwright"

# 准备目录（/workspace 由你 run 时映射进来）
RUN mkdir -p /workspace && mkdir -p /workspace-ai && chmod 1777 /workspace-ai

WORKDIR /workspace

COPY cmd-ai-dev.py /usr/local/bin/cmd-ai-dev
RUN chmod +x /usr/local/bin/cmd-ai-dev

ENV WORKSPACE=/workspace
ENV WORKSPACE_AI=/workspace-ai

ENTRYPOINT ["/usr/local/bin/cmd-ai-dev"]