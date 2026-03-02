## 1) 浏览文件（全量输出，带真实行号、空行也编号）

### 1.1 `nl`（推荐）
```bash
nl -ba FILE
```
- `-b a`：body 行号模式为 all（**包含空行**）
- 常用格式控制：
```bash
nl -ba -w6 -s$'\t' FILE
```
- `-w6`：行号宽度 6
- `-s '\t'`：行号与内容分隔符为制表符

### 1.2 `awk`（通用且可定制）
```bash
awk '{printf "%6d\t%s\n", NR, $0}' FILE
```
- `NR` 是文件真实行号（包括空行）
- 可按需改宽度/分隔符/前缀

---

## 2) 只看部分内容（头/尾/区间/单行），且行号仍为原文件真实行号

核心思路：**先编号，再截取输出**（这样截取后仍保留原行号）。

### 2.1 前 N 行（带原始行号）
```bash
nl -ba FILE | head -n 50
```

### 2.2 后 N 行（带原始行号）
```bash
nl -ba FILE | tail -n 50
```

### 2.3 第 M 行到第 N 行（闭区间，带原始行号）
```bash
nl -ba FILE | sed -n '20,40p'
```

### 2.4 只看第 K 行（带原始行号）
```bash
awk 'NR==42{printf "%d\t%s\n", NR, $0}' FILE
# 或
nl -ba FILE | sed -n '42p'
```

### 2.5 看多个不连续区间（带原始行号）
```bash
nl -ba FILE | sed -n '1,20p;50,80p;120,140p'
```

### 2.6 “从第 M 行开始取 L 行”（带原始行号）
```bash
nl -ba FILE | sed -n '100,149p'
```

---

## 3) 同时浏览多个文件（非交互）

### 3.1 顺序输出多个文件（每个文件从 1 开始编号，含空行）
```bash
for f in a.c b.c c.c; do
  printf '==> %s <==\n' "$f"
  nl -ba "$f"
done
```

### 3.2 用 `awk` 自动加文件头 + 每文件独立行号（含空行）
```bash
awk 'FNR==1{printf "==> %s <==\n", FILENAME}
     {printf "%6d\t%s\n", FNR, $0}' file1 file2 file3
```
- `FNR`：当前文件内行号（包含空行）
- `FILENAME`：当前文件名

### 3.3 多文件各取前 N 行（带原始行号）
```bash
for f in *.py; do
  printf '==> %s <==\n' "$f"
  nl -ba "$f" | head -n 80
done
```

---

## 4) 在**单个文件**中搜索代码（输出真实行号）

### 4.1 `grep`：输出匹配行 + 行号
```bash
grep -nH --color=always 'PATTERN' FILE
```
- `-n`：显示行号（真实行号）
- `-H`：显示文件名（单文件也可强制显示）
- `--color=always`：高亮匹配（仍是非交互输出）

常用变体：
```bash
grep -nH -F 'LITERAL' FILE          # 固定字符串（不解释正则）
grep -nH -w 'WORD' FILE             # 整词匹配
grep -nH -i 'pattern' FILE          # 忽略大小写
grep -nH -E 'regex' FILE            # ERE 正则
grep -nH -P 'regex' FILE            # PCRE（视系统支持）
```

### 4.2 输出匹配行的上下文（仍保持真实行号）
```bash
grep -nH -C 3 'PATTERN' FILE        # 前后各 3 行
grep -nH -A 10 'PATTERN' FILE       # 匹配行后 10 行
grep -nH -B 10 'PATTERN' FILE       # 匹配行前 10 行
```
> 注：`grep -n` 配合 `-A/-B/-C` 时，**上下文行也会带真实行号**（格式可能用 `-`/`:` 区分上下文与命中行）。

### 4.3 只输出匹配的行号（便于后续二次截取）
```bash
grep -n 'PATTERN' FILE | cut -d: -f1
```

---

## 5) 在目录/多文件中搜索代码（按文件名筛选 + 输出行号）

### 5.1 `grep` 递归搜索（常用）
```bash
grep -RIn --color=always 'PATTERN' DIR/
```
- `-R` 递归
- `-I` 忽略二进制文件（避免乱码）
- `-n` 行号（真实行号）
- 进一步排除/包含：
```bash
grep -RIn --exclude='*.min.js' --exclude-dir='.git' 'PATTERN' DIR/
grep -RIn --include='*.py' 'PATTERN' DIR/
```

### 5.2 `find + grep`（精确控制哪些文件参与搜索）
```bash
find DIR -type f -name '*.c' -print0 | xargs -0 grep -nH 'PATTERN'
```
- `-print0` + `xargs -0`：正确处理文件名含空格/特殊字符
- `grep -nH`：输出 `file:line:content`

也可用 `-exec ... {} +`（不经 xargs）：
```bash
find DIR -type f -name '*.py' -exec grep -nH 'PATTERN' {} +
```

---

## 6) 根据文件名搜索/定位代码文件（再结合浏览/搜索）

### 6.1 `find` 按文件名匹配
```bash
find DIR -type f -name '*test*'
find DIR -type f \( -name '*.c' -o -name '*.h' \)
find DIR -type f -iname '*readme*'   # 不区分大小写
```

### 6.2 将“文件名筛选”与“带行号浏览”组合
```bash
find DIR -type f -name '*.py' -print0 |
  xargs -0 -I{} sh -c 'printf "==> %s <==\n" "$1"; nl -ba "$1" | head -n 80' sh {}
```

---

## 7) 常见“组合式”查看需求（严谨、可拼接）

### 7.1 先找匹配行号，再抽取某个范围（保持真实行号）
例如：先找到匹配的行号，再取“匹配行前后各 20 行”（适合进一步加工）：
```bash
line=$(grep -n 'PATTERN' FILE | head -n 1 | cut -d: -f1)
start=$((line-20)); [ "$start" -lt 1 ] && start=1
end=$((line+20))
nl -ba FILE | sed -n "${start},${end}p"
```

### 7.2 输出文件总行数（辅助确定范围）
```bash
wc -l FILE
```

### 7.3 显示不可见字符（如制表符、行尾空格）同时带行号  
（`sed -n 'l'` 会以可见形式显示行尾/特殊字符）
```bash
nl -ba FILE | sed -n '1,80p' | sed -n 'l'
```
> 注：`sed -n 'l'` 会对行做“可视化转义展示”；用于排查不可见字符很有用。

---

## 8) 关于“行号与空行编号”的关键注意点

- **满足“空行也编号 + 真实行号”**：优先使用  
  - `nl -ba FILE`  
  - 或 `awk '{printf "%d\t%s\n", NR, $0}' FILE`
- `cat -n FILE`：默认**不会**给空行编号