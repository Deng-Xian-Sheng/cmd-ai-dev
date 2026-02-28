FROM docker.cnb.cool/cnb/cool/default-dev-env/dockerfile-caches:ba8c7bf15bfb07dec83820d9a3878fa3134a09a3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv python3-dev \
  bash git ca-certificates curl gnupg sudo ffmpeg \
  build-essential pkg-config \
  less man-db \
  procps psmisc lsof strace \
  iproute2 iputils-ping dnsutils netcat-openbsd \
  openssh-client rsync wget \
  jq tree \
  unzip zip xz-utils bzip2 tar \
  vim-tiny nano tmux bash-completion \
  openssl \
  locales tzdata \
  \
  # browser gui + vnc/novnc + fonts + magic
  libmagic1 \
  xvfb x11vnc novnc websockify fluxbox dbus-x11 \
  fonts-noto-cjk fonts-wqy-zenhei \
  && rm -rf /var/lib/apt/lists/*

# Install Google Chrome stable (amd64)
RUN set -eux; \
  wget -qO- https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /usr/share/keyrings/google-linux-signing-keyring.gpg; \
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list; \
  apt-get update; \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends google-chrome-stable; \
  rm -rf /var/lib/apt/lists/*; \
  mkdir -p /var/lib/bin/google/chrome; \
  ln -sf "$(command -v google-chrome-stable)" /var/lib/bin/google/chrome/google-chrome

# venv环境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 装依赖：你后续想切 http client 或换 UI 都方便
RUN python3 -m pip install -U pip setuptools wheel \
  && python3 -m pip install --no-cache-dir \
  "openai>=1.0.0" \
  "textual>=0.70.0" \
  "httpx>=0.25.0" \
  "markdownify" \
  "beautifulsoup4" \
  "lxml" \
  "playwright" \
  "requests>=2.31.0" \
  "tenacity>=8.2.0" \
  "pydantic>=2.0.0" \
  "python-dotenv>=1.0.0" \
  "pyyaml>=6.0.0" \
  "rich>=13.0.0" \
  "typer>=0.9.0" \
  "ipython>=8.0.0" \
  "tqdm>=4.66.0" \
  "pytest>=7.0.0" \
  "pytest-cov>=4.0.0" \
  "ruff>=0.1.0" \
  "black>=23.0.0" \
  "mypy>=1.0.0" \
  "pre-commit>=3.0.0" \
  "python-magic>=0.4.27"

# 准备目录（/workspace 由你 run 时映射进来）
RUN mkdir -p /workspace /workspace-ai

COPY cmd-ai-dev.py /usr/local/bin/cmd-ai-dev
RUN chmod +x /usr/local/bin/cmd-ai-dev

COPY look_imgs.py /usr/local/bin/look_imgs
COPY start-gui /usr/local/bin/start-gui
RUN chmod +x /usr/local/bin/look_imgs /usr/local/bin/start-gui

ENV WORKSPACE=/workspace
ENV WORKSPACE_AI=/workspace-ai

# 构建时传入宿主机 uid/gid：--build-arg UID=$(id -u) --build-arg GID=$(id -g)
ARG UID=1000
ARG GID=1000

RUN set -eux; \
    # 创建/调整 group: ai
    if getent group ai >/dev/null; then \
      groupmod -g "${GID}" ai; \
    else \
      groupadd -g "${GID}" ai; \
    fi; \
    \
    # 创建/调整 user: ai
    if id -u ai >/dev/null 2>&1; then \
      usermod -u "${UID}" -g "${GID}" -d /home/ai -s "$(command -v zsh)" ai; \
    else \
      useradd -m -d /home/ai -s "$(command -v zsh)" -u "${UID}" -g "${GID}" ai; \
    fi; \
    \
    # 确保家目录属主正确
    mkdir -p /home/ai; \
    chown -R "${UID}:${GID}" /home/ai; \
    \
    # sudo 权限（容器里通常给 NOPASSWD，否则你还得设置密码）
    echo "ai ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ai; \
    chmod 0440 /etc/sudoers.d/ai

RUN chown -R "${UID}:${GID}" /workspace /workspace-ai /opt/venv

ENV HOME=/home/ai

USER ai
WORKDIR /workspace
SHELL ["/usr/bin/zsh", "-lc"]

ENTRYPOINT ["/usr/local/bin/cmd-ai-dev"]