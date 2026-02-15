# cc-proxy

本地代理服务，将 Claude Code 的 Anthropic API 请求转换为 OpenAI 兼容格式，转发到你的中转站。

## 工作原理

```
Claude Code  ──(Anthropic API)──>  cc-proxy (localhost:18081)  ──(OpenAI API)──>  中转站
```

cc-proxy 伪装为 Anthropic Messages API 端点，接收 Claude Code 的请求后：

1. 将 Anthropic 请求格式转换为 OpenAI Chat Completions 格式
2. 转发到你的中转站
3. 将中转站的 OpenAI 格式响应转换回 Anthropic 格式
4. 返回给 Claude Code

## 支持的功能

- 基本文本消息和多轮对话
- System Prompt
- 图片消息（base64）
- Tool Use / Function Calling
- Thinking / Reasoning（`reasoning_content` 字段）
- 流式响应（SSE）
- 非流式响应
- 错误格式转换

## 快速开始

### 1. 配置

复制配置模板并填入你的中转站信息：

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`：

```yaml
server:
  host: "0.0.0.0"
  port: 18081

upstream:
  base_url: "https://your-relay.example.com"  # 中转站地址
  api_key: "sk-xxx"                            # 中转站 API Key
  timeout: 300                                 # 请求超时（秒）
```

### 2. 启动

提供三种启动方式，任选其一：

#### uv（推荐）

```bash
uv run main.py
```

#### pip

```bash
pip install -r requirements.txt
python main.py
```

#### Docker

```bash
# 先编辑好 config.yaml，然后：
docker compose up -d
```

### 3. 配置 Claude Code

设置环境变量，让 Claude Code 将请求发到本地代理：

```bash
export ANTHROPIC_BASE_URL=http://localhost:18081
export ANTHROPIC_API_KEY=any-value
```

`ANTHROPIC_API_KEY` 可以填任意值，代理不校验，但 Claude Code 要求必须设置。

## 配置说明

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `server.host` | 代理监听地址 | `0.0.0.0` |
| `server.port` | 代理监听端口 | `18081` |
| `upstream.base_url` | 中转站地址 | 无，必填 |
| `upstream.api_key` | 中转站 API Key | 无，必填 |
| `upstream.timeout` | 请求超时时间（秒） | `300` |

## 运行测试

```bash
pytest tests/ -v
```
