# OpenClaw 开发容器环境详情

> 调研日期：2026-03-04
> 容器名称：`claw`
> 创建时间：2026-03-02

---

## 1. 宿主机概况

| 项目 | 值 |
|---|---|
| CPU | AMD EPYC 9K65 192-Core Processor（分配 8 vCPU） |
| 内存 | 32 GB |
| 磁盘 | 200 GB（已用 11G，剩余 178G） |
| GPU | **无** |
| 操作系统 | Ubuntu（内核 6.8.0-71-generic） |

这是一台腾讯云或类似云厂商的 VPS，使用 AMD EPYC 9K65 服务器级 CPU。

---

## 2. 容器资源配置

### 2.1 资源限制

| 项目 | 设置 | 含义 |
|---|---|---|
| CPU 配额 | `max`（无限制） | 可用全部 8 vCPU |
| 内存上限 | `max`（无限制） | 可用全部 32 GB |
| PID 上限 | 37356 | 容器内最多进程数 |
| Swap | 1.9 GB（宿主机级别） | 共享宿主机 swap |
| 特权模式 | 否 | 标准安全策略 |
| Capabilities | 默认（无额外添加或删除） | 标准权限 |
| 重启策略 | `no` | 容器退出后不自动重启 |

**结论：容器没有设置任何 CPU/内存限制，可以使用宿主机的全部 8 核 32G 资源。**

### 2.2 存储

| 项目 | 值 |
|---|---|
| 存储驱动 | overlay2 |
| 容器可写层已用 | ~1.46 GB |
| Volume 挂载 | **无**（数据全在容器内） |
| 可用磁盘空间 | ~178 GB |

**重要提醒：容器没有挂载任何宿主机目录。如果容器被删除（`docker rm`），所有数据将丢失。建议定期备份 `/root/.openclaw/` 目录到宿主机。**

### 2.3 网络

| 项目 | 值 |
|---|---|
| 网络模式 | bridge（默认） |
| 容器 IP | 172.17.0.2/16 |
| 网关 | 172.17.0.1 |
| 端口映射 | 容器 18789 → 宿主机 18789 |
| DNS | 183.60.83.19, 183.60.82.98（腾讯云 DNS） |

### 2.4 外部服务连通性

| 服务 | 地址 | 状态 | 延迟 |
|---|---|---|---|
| DeepSeek API | api.deepseek.com | 可达（401 = 需认证） | 50ms |
| HFAI 代理 | proxy-public.high-five-ai.xyz | 可达（403 = 需认证） | 2.3s |
| GitHub | github.com | 可达（403 = 无 token） | 460ms |
| npm Registry | registry.npmjs.org | 正常 (200) | 1.7s |
| 飞书 API | open.feishu.cn | 正常 (200) | 75ms |

**所有关键外部服务均可访问**，飞书 API 延迟极低（同区域）。Google 不可达（符合国内网络环境预期）。

---

## 3. 系统环境

### 3.1 操作系统

| 项目 | 值 |
|---|---|
| 基础镜像 | `node:bookworm` |
| 发行版 | Debian GNU/Linux 12 (bookworm) |
| 内核 | 6.8.0-71-generic (x86_64) |
| 架构 | x86_64 (amd64) |
| 包管理 | apt/dpkg（已安装约 420 个包） |

### 3.2 已安装的编程语言和运行时

| 语言/运行时 | 版本 | 备注 |
|---|---|---|
| **Node.js** | v25.5.0 | 主运行时，最新版 |
| **npm** | 11.8.0 | 包管理 |
| **Yarn** | 1.22.22 | 备用包管理 |
| **npx** | 11.8.0 | npm 包执行器 |
| **Python 3** | 3.11.2 | 有标准库，**无 pip**（需手动安装） |
| **GCC/G++** | 12.2.0 | C/C++ 编译器 |
| **Make** | 4.3 | 构建工具 |
| **Git** | 2.39.5 | 版本控制 |

**未安装的语言：** Java、Go、Rust、Ruby、PHP

### 3.3 已安装的工具

| 工具 | 版本/说明 |
|---|---|
| curl | 7.88.1（支持 OpenSSL、brotli、zstd、HTTP/2） |
| wget | 1.21.3 |
| socat | 1.7.4.4（用于端口转发） |
| nano | 7.2（文本编辑器） |
| openssh-client | 9.2p1（可 SSH 到其他机器） |
| openssl | 3.0.18 |
| tar/gzip/bzip2/unzip | 标准归档工具 |
| mercurial | 6.3.2（版本控制，Python 包形式安装） |

**未安装但可能需要的工具：** vim、tmux、screen、jq、docker-cli、htop

### 3.4 Python 标准库能力

虽然没有 pip，但 Python 3.11 的标准库已经包含：

- `json` — JSON 处理
- `csv` — CSV 处理
- `sqlite3` — 本地数据库
- `http.server` — 简易 HTTP 服务器
- `urllib` — HTTP 请求
- `asyncio` — 异步编程
- `subprocess` — 进程管理
- `pathlib` — 路径操作
- `re` — 正则表达式

如需安装 pip：`apt-get update && apt-get install -y python3-pip`

---

## 4. OpenClaw 安装信息

| 项目 | 值 |
|---|---|
| 版本 | 2026.2.26 |
| 最新可用版本 | 2026.3.2 |
| 安装方式 | npm 全局安装 |
| 安装路径 | /usr/local/lib/node_modules/openclaw/ |
| 安装大小 | ~759 MB |
| 入口文件 | /usr/local/bin/openclaw → openclaw.mjs |
| 已安装插件 | @openclaw/feishu (2026.3.1) |

---

## 5. 能做什么 / 不能做什么

### 5.1 适合做的任务

| 任务类型 | 可行性 | 说明 |
|---|---|---|
| **飞书集成和交互** | 非常适合 | 飞书插件已安装，API 延迟极低 (75ms) |
| **Node.js/JavaScript 开发** | 非常适合 | Node.js v25.5 + npm + Yarn 齐全 |
| **Web 爬虫/数据采集** | 适合 | curl/wget/Node.js/Python 都可用 |
| **文本处理/数据分析** | 适合 | Python 标准库 + Node.js |
| **API 调用/集成** | 适合 | 网络畅通，curl/Node.js 都可用 |
| **小型 Web 服务** | 适合 | Node.js 或 Python http.server |
| **C/C++ 编译** | 适合 | GCC 12.2 完整工具链 |
| **Shell 脚本自动化** | 适合 | Bash + 常用工具齐全 |
| **SQLite 数据库** | 适合 | Python sqlite3 内置 |
| **LLM Agent 任务编排** | 非常适合 | OpenClaw 核心能力 |
| **定时任务/Cron** | 适合 | OpenClaw 内置 cron 系统 |
| **SSH 远程管理** | 适合 | openssh-client 已安装 |

### 5.2 需要额外安装才能做的任务

| 任务类型 | 需要安装 | 安装命令 |
|---|---|---|
| Python 第三方库 | pip | `apt-get update && apt-get install -y python3-pip` |
| Python 数据科学 | numpy/pandas/scipy | 先装 pip，再 `pip3 install numpy pandas` |
| Docker-in-Docker | docker-cli | 不推荐，建议在宿主机操作 |
| Java 开发 | JDK | `apt-get install -y default-jdk` |
| Go 开发 | Go | 手动下载安装 |
| Rust 开发 | Rust | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

### 5.3 不适合/无法做的任务

| 任务类型 | 原因 |
|---|---|
| **GPU 计算 / 深度学习训练** | 无 GPU，纯 CPU 机器 |
| **本地运行大模型推理** | 无 GPU，内存 32G 也偏小 |
| **大规模数据处理 (>100GB)** | 磁盘 178G 剩余，且无 volume 持久化 |
| **生产级高可用服务** | 容器无重启策略，无 volume，单点部署 |
| **需要 Google 服务的任务** | 网络不通 Google（国内环境） |
| **图形界面 / GUI 应用** | 无显示设备，纯终端环境 |
| **视频/音频实时处理** | 无 GPU 加速，CPU 处理效率低 |

### 5.4 资源余量评估

```
CPU:    8 核 AMD EPYC 9K65（当前几乎空闲）    → 计算余量充足
内存:   32 GB（当前使用 ~3.2G，剩余 ~27G）     → 内存余量充足
磁盘:   200 GB（当前使用 ~11G，剩余 ~178G）    → 存储余量充足
网络:   国内带宽，飞书/DeepSeek 延迟优秀         → 网络条件良好
```

**总结：这是一台配置不错的纯 CPU 云服务器，非常适合做 Agent 任务编排、API 调用、飞书集成、Node.js 开发和轻量级数据处理。不适合做 GPU 相关的深度学习任务。**

---

## 6. 容器维护建议

### 6.1 数据备份（重要）

容器没有 volume 挂载，数据全在容器可写层中。建议定期备份：

```bash
# 备份 OpenClaw 配置和数据
sudo docker cp claw:/root/.openclaw /home/ubuntu/zenghui/openclaw/backup/

# 或者用 tar 打包
sudo docker exec claw tar czf /tmp/openclaw-backup.tar.gz /root/.openclaw/
sudo docker cp claw:/tmp/openclaw-backup.tar.gz /home/ubuntu/zenghui/openclaw/backup/
```

### 6.2 容器重启策略

当前重启策略为 `no`，宿主机重启后容器不会自动启动。如需改为自动重启：

```bash
sudo docker update --restart unless-stopped claw
```

### 6.3 更新 OpenClaw

```bash
# 容器内更新
sudo docker exec claw npm install -g openclaw@latest

# 更新后重启服务
bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh
bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh
```

### 6.4 安装额外工具

容器基于 Debian bookworm，可以自由使用 apt 安装软件：

```bash
sudo docker exec claw bash -c "apt-get update && apt-get install -y <package>"
```

注意：通过 `apt-get` 安装的软件在容器重建后会丢失。如果需要持久化，考虑构建自定义 Dockerfile。
