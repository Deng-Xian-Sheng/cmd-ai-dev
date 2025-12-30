# cmd-ai-dev

一个大道至简的命令行 AI 编程工具：在 Docker 容器里让模型通过“只提供命令行”完成编程任务，并用左右分栏 TUI 提升 16:9 屏幕下的可用性。

## 安装与使用

### 方式 A：Docker 构建并运行

在项目目录下：

```bash
docker build -t cmd-ai-dev:latest .
```

在你的项目目录（要让 AI 修改的代码库目录）运行：

```bash
docker run -it --rm \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --network host \
  -e OPENAI_API_KEY="你的key" \
  -e OPENAI_MODEL="你的模型名" \
  -e OPENAI_BASE_URL="你的base_url(可选，OpenAI-compatible 时用)" \
  -v "$PWD:/workspace" \
  cmd-ai-dev:latest
```

启动后：
- 左侧：模型输出 / 命令执行记录
- 右侧：多行输入框（适合粘贴长文本）
- 快捷键（终端兼容性优先）：
  - `Ctrl+S` 发送
  - `Ctrl+T` 停止命令链
  - `Ctrl+R` 重置会话
  - `Ctrl+Q` 退出  
  也提供按钮：发送 / 停止 / 重置 / 退出

### 方式 B：宿主机一条命令启动（推荐）

把下面内容粘贴到 `~/.bashrc`，然后 `source ~/.bashrc`：

```bash
cmd_ai_dev() {
  set -e

  # ====== 你需要填的配置 ======
  local OPENAI_API_KEY="填你的key"
  local OPENAI_BASE_URL="填你的base_url（可留空）"
  local OPENAI_MODEL="填你的模型名"
  local IMAGE_NAME="${CMD_AI_DEV_IMAGE:-cmd-ai-dev:latest}"
  # ===========================

  local proj_dir="${1:-$PWD}"
  if [ ! -d "$proj_dir" ]; then
    echo "目录不存在：$proj_dir" >&2
    return 1
  fi
  proj_dir="$(cd "$proj_dir" && pwd)"

  local proj_base parent_base
  proj_base="$(basename "$proj_dir")"
  parent_base="$(basename "$(dirname "$proj_dir")")"

  local cname="cmd-ai-dev-${parent_base}-${proj_base}"
  cname="$(echo "$cname" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_.-]/-/g')"

  local base_url_args=()
  if [ -n "$OPENAI_BASE_URL" ]; then
    base_url_args=(-e "OPENAI_BASE_URL=$OPENAI_BASE_URL")
  fi

  if docker container inspect "$cname" >/dev/null 2>&1; then
    if [ "$(docker inspect -f '{{.State.Running}}' "$cname" 2>/dev/null)" = "true" ]; then
      docker exec -it \
        -u "$(id -u):$(id -g)" \
        -e "OPENAI_API_KEY=$OPENAI_API_KEY" \
        "${base_url_args[@]}" \
        -e "OPENAI_MODEL=$OPENAI_MODEL" \
        "$cname" cmd-ai-dev
    else
      docker start -ai "$cname"
    fi
  else
    docker run -it --name "$cname" --network host \
      --user "$(id -u):$(id -g)" \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      -e "OPENAI_API_KEY=$OPENAI_API_KEY" \
      "${base_url_args[@]}" \
      -e "OPENAI_MODEL=$OPENAI_MODEL" \
      -v "$proj_dir:/workspace" \
      "$IMAGE_NAME"
  fi
}

alias cmd-ai-dev=cmd_ai_dev
```

之后你就可以：

```bash
cmd-ai-dev
# 或指定目录
cmd-ai-dev /path/to/your/project
```

## 设计与协议

### 容器内目录约定
- `/workspace`：用户项目目录（映射到宿主机）
- `/workspace-ai`：AI 工作目录（默认不映射；用于临时脚本、日志、中间产物，避免污染仓库）
- `/workspace-ai/session.json`：会话上下文（单会话）
- `/workspace-ai/cmdout_*.log`：命令输出落盘日志
- `/workspace-ai/transcript.log`：左侧显示内容的完整记录（便于复制）

### 工具调用格式（不依赖函数调用能力）
模型需要执行命令时输出：

```
<cmd>
<time_out>60</time_out>
这里是要执行的命令（可多行）
</cmd>
这里是模型的回答（可选）
```

工具执行后回给模型：

```
<cmdout>
[exit=... timeout=0/1 interrupted=0/1]
这里是命令输出（可能截断）
</cmdout>
这里是用户的交互（可选）
```

## 特点

- 单文件实现：核心逻辑集中在一个 `cmd-ai-dev.py`，没有复杂的项目结构。
- 依赖的 AI 模型易于更改：对 OpenAI-compatible 接口友好，封装层便于替换为自发 HTTP 请求。
- 给模型高灵活性但不污染系统/项目：
  - 模型可通过命令行做几乎任何事（读写文件、跑脚本、安装包等）
  - 但运行在 Docker 容器中，宿主机更安全
  - AI 的临时产物默认落在 `/workspace-ai`，避免污染你的 Git 仓库
- 工具极简：只把“命令行”作为工具提供给模型，不引入可能受限或陌生的工具体系。
- 不依赖函数调用：自定义 `<cmd>` / `<cmdout>` 格式，即使模型不支持 function calling 也能工作。
- 左右分栏布局：模型输出在左、用户输入在右，对 16:9 屏幕更友好，适合长输出与长输入并存的场景。
- 支持 Markdown 渲染：模型的叙述输出会以 Markdown 方式渲染（`<cmd>` / `<cmdout>` 仍保持工具样式）。

## 已知缺陷

- 终端兼容性：不同终端对 TUI、IME（中文标点等）、选中复制、组合键的支持差异很大。
  - 为降低风险，提供按钮操作与备选快捷键，并将左侧内容落盘到 `/workspace-ai/transcript.log` 作为复制兜底。

## 致谢

本项目在编写过程中参考并借助了 GPT-5.2-High 生成的代码建议与改进方案。