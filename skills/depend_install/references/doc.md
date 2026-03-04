## 1) APT（Debian/Ubuntu）依赖安装

> 脚本/自动化场景通常优先用 `apt-get`（比 `apt` 更稳定）。  
> 非交互常用：`-y` + `DEBIAN_FRONTEND=noninteractive`。  
> 更新索引建议先 `apt-get update`。

### 1.1 更新包索引 / 升级系统
```bash
sudo apt-get update
sudo apt-get -y upgrade              # 升级已安装包（不处理依赖变更较大的情况）
sudo apt-get -y dist-upgrade          # 允许更复杂的依赖变更（旧系统常用）
```

### 1.2 安装单个/多个包（非交互）
```bash
sudo apt-get update
sudo apt-get install -y pkg1 pkg2 pkg3
```

常用附加选项：
```bash
sudo apt-get install -y --no-install-recommends pkg1 pkg2
# --no-install-recommends：避免安装“推荐”包（更精简，容器常用）
```

### 1.3 安装指定版本（pin version）
```bash
apt-cache policy pkg
sudo apt-get install -y pkg=1.2.3-1ubuntu4
```

### 1.4 从文件批量安装（包名列表）
假设 `packages.txt` 每行一个包名：
```bash
sudo apt-get update
xargs -a packages.txt sudo apt-get install -y
```

### 1.5 搜索/查询
```bash
apt-cache search keyword
apt-cache show pkg
dpkg -l | grep -E '^ii\s+pkg'
dpkg -L pkg                     # 列出包安装了哪些文件
```

### 1.6 卸载/清理
```bash
sudo apt-get remove -y pkg            # 卸载包（保留配置）
sudo apt-get purge -y pkg             # 卸载包（删除配置）
sudo apt-get autoremove -y            # 清理不再需要的依赖
sudo apt-get clean                    # 清理已下载的 .deb 缓存
sudo apt-get autoclean                # 清理过期缓存
```

### 1.7 处理“配置提示/交互”（强制非交互）
```bash
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y pkg
```

某些场景（配置文件冲突）可用 dpkg 选项（谨慎）：
```bash
sudo DEBIAN_FRONTEND=noninteractive apt-get \
  -o Dpkg::Options::="--force-confnew" \
  -o Dpkg::Options::="--force-confdef" \
  install -y pkg
```

### 1.8 添加源（典型：PPA / 自定义源）
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 添加 keyring（推荐做法：/etc/apt/keyrings）
curl -fsSL https://example.com/repo.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/example.gpg

# 添加源列表
echo "deb [signed-by=/etc/apt/keyrings/example.gpg] https://example.com/debian stable main" \
  | sudo tee /etc/apt/sources.list.d/example.list > /dev/null

sudo apt-get update
sudo apt-get install -y somepkg
```

---

## 2) pip（Python）依赖安装

> 自动化建议：优先使用 `python -m pip`，避免 `pip` 指向错误解释器。  
> 非交互：`--no-input`。  
> 强烈建议在虚拟环境中安装（venv/conda/poetry等）。

### 2.1 创建并使用 venv（推荐）
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2.2 安装单个/多个包
```bash
python -m pip install pkg
python -m pip install pkg1 pkg2
```

### 2.3 安装指定版本 / 版本范围
```bash
python -m pip install 'requests==2.31.0'
python -m pip install 'urllib3>=2.0,<3.0'
```

### 2.4 从 requirements.txt 批量安装（最常用）
```bash
python -m pip install -r requirements.txt
```

常用参数：
```bash
python -m pip install --no-input -r requirements.txt
python -m pip install --upgrade -r requirements.txt
python -m pip install --no-cache-dir -r requirements.txt
```

### 2.5 生成依赖锁定（仅 pip 常用：freeze）
```bash
python -m pip freeze > requirements.lock
```

### 2.6 安装本地项目（当前目录）/ 可编辑安装
```bash
python -m pip install .
python -m pip install -e .            # editable（开发态）
```

### 2.7 使用镜像源/代理（示例）
```bash
python -m pip install -i https://pypi.org/simple pkg
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://mirror/simple pkg
```

### 2.8 下载离线包（wheel/源码包），再离线安装
```bash
python -m pip download -d ./wheels -r requirements.txt
python -m pip install --no-index --find-links=./wheels -r requirements.txt
```

### 2.9 卸载/查询
```bash
python -m pip uninstall -y pkg
python -m pip show pkg
python -m pip list
python -m pip check                # 检查依赖冲突
```

---

## 3) Go（Golang）依赖安装（modules）

> Go 1.17+ 推荐：项目依赖由 `go.mod/go.sum` 管理；  
> 工具安装用 `go install module@version`；  
> 不再推荐在模块模式下用 `go get` 来安装二进制工具（`go get` 主要用于调整依赖）。

### 3.1 初始化模块（新项目）
```bash
go mod init example.com/myproj
```

### 3.2 安装/下载项目依赖（根据 go.mod）
```bash
go mod download          # 下载模块到本地缓存（不改 go.mod）
go mod tidy              # 增删依赖使 go.mod/go.sum 与代码一致（会改文件）
go build ./...           # 构建时也会自动拉取缺失依赖
```

### 3.3 添加/升级某个依赖版本（项目依赖）
```bash
go get example.com/pkg@v1.2.3
go get example.com/pkg@latest
go get -u ./...          # 升级所有依赖到较新版本（可能引入破坏性变化，谨慎）
```

### 3.4 安装 Go 工具（生成可执行文件，推荐写法）
```bash
go install golang.org/x/tools/cmd/goimports@latest
go install golang.org/x/tools/cmd/goimports@v0.20.0
```

> 二进制默认安装到：
> - `GOBIN`（若设置）
> - 否则 `$GOPATH/bin`

### 3.5 使用代理/私有仓库常见环境变量
```bash
go env -w GOPROXY=https://proxy.golang.org,direct
go env -w GOSUMDB=sum.golang.org

# 私有模块（示例：公司域名）
go env -w GOPRIVATE='git.example.com,github.com/myorg/*'
go env -w GONOSUMDB='git.example.com,github.com/myorg/*'
```

### 3.6 清理/诊断
```bash
go clean -modcache       # 清空模块下载缓存（会导致下次重新下载）
go list -m all           # 列出所有模块依赖
go mod graph             # 依赖图
```

---

## 4) Rust（cargo）依赖安装

> Rust 项目依赖由 `Cargo.toml` 和 `Cargo.lock` 管理。  
> `cargo build`/`cargo run` 会自动拉取依赖；  
> 工具（可执行程序）安装用 `cargo install`。  
> Rust 工具链安装/切换通常由 `rustup` 管理。

### 4.1 安装 Rust 工具链（rustup）
```bash
# 安装 rustup（非交互）
curl -fsSL https://sh.rustup.rs | sh -s -- -y

# 安装/切换工具链
rustup toolchain install stable
rustup default stable
rustup update
```

### 4.2 项目依赖：拉取/构建（根据 Cargo.toml/Cargo.lock）
```bash
cargo fetch              # 仅下载依赖
cargo build              # 构建并自动下载依赖
cargo build --release
```

### 4.3 添加依赖（编辑 Cargo.toml）
推荐使用 cargo-edit（需要先安装）：
```bash
cargo install cargo-edit
cargo add serde
cargo add serde --features derive
cargo add tokio --features full
cargo add anyhow@1.0.86
```

若不使用 `cargo add`，则直接修改 `Cargo.toml` 后：
```bash
cargo build
```

### 4.4 升级依赖
```bash
cargo update                 # 根据 Cargo.toml 允许范围更新 Cargo.lock
cargo update -p serde        # 仅更新某个 crate
```

### 4.5 安装 Rust CLI 工具（二进制）
```bash
cargo install ripgrep
cargo install ripgrep --version 14.1.0
cargo install --locked some_tool     # 强制使用 Cargo.lock（更可复现）
```

### 4.6 使用 git/path 依赖（写入 Cargo.toml 的典型形式）
```toml
[dependencies]
mycrate = { git = "https://github.com/user/mycrate", rev = "abcdef123" }
other  = { path = "../other" }
```
修改后执行：
```bash
cargo build
```

### 4.7 清理
```bash
cargo clean
```

---

## 5) “可复现/自动化”常用组合模板（便于提示词）

### 5.1 Ubuntu/Debian：非交互安装系统依赖（精简）
```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl git
```

### 5.2 Python：venv + requirements 安装（无交互）
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install --no-input -r requirements.txt
python -m pip check
```

### 5.3 Go：按 go.mod 下载 + 整理（CI 常用）
```bash
go mod download
go mod tidy
go test ./...
```

### 5.4 Rust：拉取依赖 + 构建（CI 常用）
```bash
cargo fetch
cargo build --locked
cargo test --locked
```