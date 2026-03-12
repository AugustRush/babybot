# BabyBot

基于 agentscope 1.0+ 的多 Agent 协同系统，支持统一的 JSON 配置。

## 快速开始

```bash
# 安装依赖
uv sync

# 配置 (编辑 config.json)
cp config.json.example config.json
# 编辑 config.json 填入 API key 和资源配置

# 运行
uv run babybot

# 运行飞书通道
uv run babybot --channel feishu
```

## 配置说明

所有配置都在 `config.json` 中，包括：

### 1. 模型配置

```json
{
  "model": {
    "model_name": "deepseek-ai/DeepSeek-V3.2",
    "api_key": "your-api-key",
    "api_base": "https://api-inference.modelscope.cn/v1",
    "temperature": 0.7,
    "max_tokens": 2048
  }
}
```

### 飞书通道配置

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "app_id": "cli_xxx",
      "app_secret": "xxx",
      "encrypt_key": "",
      "verification_token": "",
      "group_policy": "mention",
      "reply_mode": "chat"
    }
  }
}
```

- `group_policy`:
  - `mention`: 群聊中仅被 @ 时响应
  - `open`: 群聊中所有消息都响应
- `reply_mode`:
  - `chat`: 回复到当前会话
  - `p2p`: 优先按发送者 `open_id` 私聊回复

### 2. 工具组配置

```json
{
  "resources": {
    "tool_groups": {
      "search": {
        "active": true,
        "description": "搜索工具",
        "notes": "用于信息检索"
      },
      "code": {
        "active": false,
        "description": "代码工具"
      }
    }
  }
}
```

### 3. MCP 服务配置

**HTTP MCP 服务：**

```json
{
  "resources": {
    "mcp_servers": {
      "gaode_map": {
        "type": "http",
        "url": "https://mcp.amap.com/mcp?key=YOUR_KEY",
        "transport": "streamable_http",
        "group_name": "map_services",
        "active": false
      }
    }
  }
}
```

**StdIO MCP 服务（如 Playwright）：**

```json
{
  "resources": {
    "mcp_servers": {
      "playwright": {
        "type": "stdio",
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
        "group_name": "browser",
        "active": false
      }
    }
  }
}
```

### 4. 自定义工具配置

```json
{
  "resources": {
    "custom_tools": {
      "web_search": {
        "module": "tools.search",
        "function": "web_search",
        "group_name": "search",
        "preset_kwargs": {
          "api_key": "${SEARCH_API_KEY}"
        }
      }
    }
  }
}
```

### 5. Agent Skills 配置

```json
{
  "resources": {
    "agent_skills": {
      "data_analysis": {
        "directory": "skills/data_analysis",
        "description": "数据分析技能"
      }
    }
  }
}
```

## 完整配置示例

查看 `config.json.example` 获取完整配置模板。

## 环境变量

在配置中使用 `${VAR_NAME}` 从环境变量读取：

```json
{
  "custom_tools": {
    "search": {
      "preset_kwargs": {
        "api_key": "${SEARCH_API_KEY}"
      }
    }
  }
}
```

## 架构

```
config.json (统一配置)
├── model (模型配置)
└── resources (资源配置)
    ├── tool_groups (工具组)
    ├── mcp_servers (MCP 服务)
    ├── custom_tools (自定义工具)
    └── agent_skills (技能)
```

## 项目结构

```
babybot/
├── babybot/
│   ├── config.py         # 配置管理
│   ├── resource.py       # 资源管理器
│   ├── orchestrator.py   # 主 Agent
│   └── worker.py         # 子 Agent
├── tools/                # 自定义工具
├── config.json           # 配置文件
├── config.json.example   # 配置示例
└── README.md
```

## 特性

- ✅ **统一配置** - 所有配置在 config.json 中
- ✅ **JSON 格式** - 更好的可读性
- ✅ **环境变量** - 支持 ${VAR} 语法
- ✅ **Handoffs 模式** - 官方多 Agent 协同
- ✅ **工具组管理** - 动态激活/停用
- ✅ **MCP 支持** - 集成 MCP 服务
- ✅ **Agent Skills** - 技能目录
- ✅ **资源继承** - Worker 继承工具配置

## License

MIT
