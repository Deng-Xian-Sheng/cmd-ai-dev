## 0) 总原则（写文件时常见坑）

1. **重定向 `>`/`>>` 是由当前 shell 执行**：  
   `sudo cmd > file` 并不会让重定向拥有 root 权限。需要用 `sudo tee` 或 `sudo sh -c '...'`。
2. **尽量用 `printf` 替代 `echo`**（`echo` 在不同 shell 对 `-e`、转义等行为不一致）。
3. **批量/重要修改建议：先写临时文件再原子替换**：  
   `tmp=$(mktemp); ... >"$tmp" && mv "$tmp" file`（`mv` 在同一文件系统内通常是原子的）。
4. **跨平台差异**：本文以 Linux/GNU 工具为主（如 `sed -i` 的参数在 macOS 不同）。

---

## 1) 创建文件/清空文件/写入固定内容

### 1.1 创建空文件（不存在则创建，存在则更新时间戳）
```bash
touch FILE
```

### 1.2 清空文件（truncate to zero）
```bash
: > FILE
# 或
truncate -s 0 FILE
```

### 1.3 覆盖写入一行/多行（推荐 printf）
```bash
printf '%s\n' 'line1' 'line2' 'line3' > FILE
```

### 1.4 追加写入
```bash
printf '%s\n' 'another line' >> FILE
```

---

## 2) 用 Here-Doc 写入“整段代码/配置”（最常用）

### 2.1 覆盖写入整段文本（保留原样，禁止变量展开/反斜杠转义）
```bash
cat > FILE <<'EOF'
line 1
line 2: $HOME will NOT expand
backslash \ stays
EOF
```

### 2.2 追加整段文本
```bash
cat >> FILE <<'EOF'
append block...
EOF
```

### 2.3 需要变量展开时（不加引号的 EOF）
```bash
cat > FILE <<EOF
home=$HOME
EOF
```

---

## 3) `tee`：同时写文件 + 输出到 stdout（并解决 sudo 重定向问题）

### 3.1 覆盖写入
```bash
printf '%s\n' 'content' | tee FILE > /dev/null
```

### 3.2 追加写入
```bash
printf '%s\n' 'content' | tee -a FILE > /dev/null
```

### 3.3 写入需要 root 权限的文件（推荐写法）
```bash
cat <<'EOF' | sudo tee /etc/some.conf > /dev/null
key=value
EOF
```

> 说明：`sudo tee file` 由 tee 进程以 root 打开文件，因此可写入受限路径。

---

## 4) 按“行号/区间”修改文件（插入/替换/删除）

这类操作通常用 `sed`/`awk`/`ed -s` 实现。**若你要求“基于真实行号”做编辑**，推荐用这些方式（但注意：编辑会改变后续行号）。

### 4.1 `sed`：删除第 N 行 / 删除区间
```bash
sed '42d' FILE > FILE.new && mv FILE.new FILE
sed '10,20d' FILE > FILE.new && mv FILE.new FILE
```

### 4.2 `sed`：替换第 N 行整行内容
```bash
sed '42c\NEW WHOLE LINE' FILE > FILE.new && mv FILE.new FILE
```

### 4.3 `sed`：在第 N 行前插入 / 在第 N 行后追加
```bash
# 在第 10 行前插入一行
sed '10i\INSERTED LINE' FILE > FILE.new && mv FILE.new FILE

# 在第 10 行后追加一行
sed '10a\APPENDED LINE' FILE > FILE.new && mv FILE.new FILE
```

### 4.4 `sed`：插入/追加多行（GNU sed 常见写法）
```bash
sed '10i\
line A\
line B\
line C' FILE > FILE.new && mv FILE.new FILE
```

### 4.5 `sed -i`：就地修改（可选备份扩展名）
```bash
sed -i.bak '10,20d' FILE     # 修改并生成 FILE.bak
sed -i 's/old/new/g' FILE    # 不备份（有风险）
```

---

## 5) 按“内容模式”修改（搜索/替换/条件编辑）

### 5.1 `sed`：全文替换（正则）
```bash
sed 's/OLD/NEW/g' FILE > FILE.new && mv FILE.new FILE
```

### 5.2 只替换首次匹配
```bash
sed 's/OLD/NEW/' FILE > FILE.new && mv FILE.new FILE
```

### 5.3 仅在匹配某些行时替换（地址条件）
```bash
sed '/PATTERN/s/OLD/NEW/g' FILE > FILE.new && mv FILE.new FILE
```

### 5.4 删除匹配某模式的行
```bash
sed '/PATTERN/d' FILE > FILE.new && mv FILE.new FILE
```

### 5.5 用 `perl -pi` 做就地替换（功能强，适合复杂场景）
```bash
perl -pi.bak -e 's/OLD/NEW/g' FILE
```

- 多行/跨行处理常用 slurp（整文件读入）：
```bash
perl -0777 -pe 's/BEGIN.*?END/REPLACED/sg' FILE > FILE.new && mv FILE.new FILE
```

---

## 6) 生成文件：用程序化方式输出内容（awk/python等）

### 6.1 `awk` 生成/重写文件
```bash
awk 'BEGIN{print "header"} {print NR "\t" $0}' INPUT > OUTPUT
```

### 6.2 `python` 写文件（适合复杂模板/JSON/YAML 生成）
```bash
python3 - <<'PY'
from pathlib import Path
Path("FILE").write_text("line1\nline2\n", encoding="utf-8")
PY
```

---

## 7) 同时修改多个文件（批量）

### 7.1 `find + sed -i` 批量替换（按扩展名筛选）
```bash
find DIR -type f -name '*.py' -exec sed -i.bak 's/OLD/NEW/g' {} +
```

### 7.2 `find + perl -pi` 批量替换
```bash
find DIR -type f -name '*.c' -exec perl -pi.bak -e 's/\bfoo\b/bar/g' {} +
```

### 7.3 `xargs`（配合 `-print0` 安全处理特殊文件名）
```bash
find DIR -type f -name '*.js' -print0 | xargs -0 perl -pi.bak -e 's/OLD/NEW/g'
```

---

## 8) 用“补丁”方式写/改文件

当你能生成 unified diff 时，`patch`/`git apply` 通常比一堆 `sed` 更稳健。

### 8.1 `patch`：从 diff 修改文件
```bash
patch -p0 < changes.diff
```

也可 here-doc：
```bash
patch -p0 <<'EOF'
*** oldfile.txt
--- oldfile.txt
***************
*** 1,3 ****
- old
+ new
EOF
```

### 8.2 `git apply`：在 git 仓库中应用 diff（不要求提交）
```bash
git apply changes.diff
```

常用检查：
```bash
git apply --check changes.diff
```

---

## 9) 安全写入（原子替换、权限保留、备份）

### 9.1 原子替换（推荐通用模式）
```bash
tmp=$(mktemp)
cat > "$tmp" <<'EOF'
new content
EOF
mv "$tmp" FILE
```

### 9.2 保留权限/属主（常用做法之一：先拷贝元信息再覆盖）
- 若 `FILE` 已存在且你想尽量保持权限，可先：
```bash
tmp=$(mktemp)
cat > "$tmp" <<'EOF'
new content
EOF
chmod --reference=FILE "$tmp"
chown --reference=FILE "$tmp" 2>/dev/null || true
mv "$tmp" FILE
```

### 9.3 创建文件时指定权限（适合新文件）
```bash
install -m 0644 /dev/null FILE
# 然后再写入
cat > FILE <<'EOF'
content
EOF
```

---

## 10) 常用“组合任务”模板

### 10.1 “将一段内容写成新文件（覆盖）”
```bash
cat > PATH/TO/FILE <<'EOF'
...content...
EOF
```

### 10.2 “在文件末尾追加一段内容”
```bash
cat >> PATH/TO/FILE <<'EOF'
...append...
EOF
```

### 10.3 “替换文件中的某个字符串（全局）并备份”
```bash
sed -i.bak 's/OLD/NEW/g' FILE
```

### 10.4 “删除第 M 到 N 行（生成新文件再替换，避免 sed -i 差异）”
```bash
sed 'M,Nd' FILE > FILE.new && mv FILE.new FILE
```

### 10.5 “对目录下所有 *.ext 文件做替换（带备份）”
```bash
find DIR -type f -name '*.ext' -exec sed -i.bak 's/OLD/NEW/g' {} +
```