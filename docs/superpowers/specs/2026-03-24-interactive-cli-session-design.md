# Interactive CLI Session Mode Design

## Goal

为 BabyBot 增加“按聊天会话绑定的长驻 CLI agent 会话”能力。用户通过固定控制命令进入或退出某个交互 backend；进入后，当前聊天的后续消息直接与该 backend 的长驻子进程进行实时双向交互，而不是继续走普通编排链路。

首个落地 backend 为 Claude Code，后续可平滑扩展到 Codex 等其他 CLI agent。

## Scope

本设计覆盖：

- 每个 `chat_key` 最多绑定一个活动交互会话
- `@session` 控制命令
- 通用交互会话管理层
- Claude Code backend 的首个实现
- 显式停止与可配置硬上限回收
- 会话中消息透传与状态查询

本设计暂不覆盖：

- 多 backend 并存于同一聊天
- 会话状态跨进程持久化恢复
- 在 BabyBot 侧重写 Claude/Codex 原生命令语义
- 默认启动任何交互会话

## User Experience

BabyBot 自身只占用 `@session` 前缀，避免与 backend 原生命令冲突。

首批控制命令：

- `@session start claude`
- `@session stop`
- `@session status`

行为规则：

- 默认情况下，消息继续走现有 `OrchestratorAgent` 处理链
- 用户发送 `@session start claude` 后，仅当前 `chat_key` 进入 Claude 交互模式
- 交互模式下的普通文本直接转发给 Claude backend
- 交互模式下的 `/models`、`/skill` 等命令原样透传给 Claude，不由 BabyBot 解释
- `@session ...` 在任何时候都由 BabyBot 自己解释，优先级最高
- 用户发送 `@session stop` 后，当前聊天的交互会话立即销毁
- 会话达到硬上限后自动销毁，并向用户返回一条明确提示

## Architecture

### Routing Layer

会话模式的入口放在 `OrchestratorAgent.process_task()`，而不是 `MessageBus` 或 channel 层。

原因：

- `process_task()` 已经以 `chat_key` 为核心组织状态
- 这里最适合做“普通模式 / 交互模式”的统一分流
- 能避免把业务语义耦合进通道传输层
- CLI、飞书、微信都能共用同一套逻辑

### Core Components

建议新增以下模块：

- `babybot/interactive_sessions/types.py`
  负责会话状态、backend 标识、控制命令解析结果等数据结构
- `babybot/interactive_sessions/protocols.py`
  定义 `InteractiveBackend` 协议
- `babybot/interactive_sessions/manager.py`
  负责 `chat_key -> session` 生命周期、并发保护、超时回收、状态查询
- `babybot/interactive_sessions/backends/claude.py`
  Claude Code 的 backend 适配实现
- `babybot/interactive_sessions/__init__.py`
  导出公共接口

建议修改：

- `babybot/orchestrator.py`
  增加控制命令解析、会话路由、停止与清理逻辑
- `babybot/config.py`
  增加交互会话配置项
- `config.json.example`
  暴露默认配置
- `tests/`
  增加 manager、orchestrator 路由、Claude backend 的测试

### Backend Abstraction

通用协议定义为：

- `start(chat_key, workdir, config) -> SessionHandle`
- `send(session, message) -> BackendReply`
- `stop(session, reason) -> None`
- `status(session) -> BackendStatus`

关键约束：

- Manager 不理解 Claude/Codex 私有协议
- Backend 不关心 channel，只处理会话与消息
- Orchestrator 只依赖 manager，不依赖具体 backend 实现

这样后续新增 `CodexInteractiveBackend` 时，不需要重写路由与生命周期管理。

## Claude Backend Design

### Process Model

Claude backend 为每个活动聊天创建一个独立的长驻 Claude CLI 子进程。

会话粒度采用：

- `1 chat_key -> 1 Claude process`

原因：

- 隔离最好，不会串聊天上下文
- 最符合“一个聊天就是一个连续会话”的用户模型
- 停止命令和超时回收语义最直接

### Transport Strategy

Claude backend 内部保留“驱动层”边界，首个实现优先使用 CLI 的流式输入输出协议；如果本地验证发现某些 Claude Code 工具能力在该模式下缺失，再退回 PTY 驱动，但不改变 manager 和 orchestrator 接口。

首版设计约束：

- backend 需要支持长驻进程
- backend 需要支持逐条发送用户消息
- backend 需要支持持续读取模型/工具输出
- backend 需要支持完整 Claude Code 工作目录能力

无论采用哪种驱动，`ClaudeInteractiveBackend` 对外暴露的仍是统一的 `send()` / `stop()` / `status()` 接口。

### Environment Isolation

Claude backend 不能直接继承宿主环境的默认持久化目录。

必须显式隔离：

- 会话运行目录
- 状态目录
- 临时目录

目标：

- 避免写入 `~/.claude*` 导致 sandbox/CI 权限错误
- 让 backend 行为在本地、测试、受限环境下更稳定
- 会话销毁时能够完整清理临时状态

## Session Lifecycle

### Start

当当前聊天收到 `@session start claude` 时：

1. 检查该 `chat_key` 是否已有活动会话
2. 若已有同 backend 会话，直接返回“已在会话中”
3. 若已有其他 backend 会话，先拒绝并要求先 `@session stop`
4. 若无活动会话，懒启动 Claude backend

### Message Flow

当某个 `chat_key` 存在活动会话时：

1. `@session ...` 交给控制层处理
2. 其他消息直接发送给当前 backend
3. backend 返回的文本作为当前聊天回复
4. backend 异常退出时，manager 注销该会话并返回错误提示

### Stop

停止条件有两类：

- 用户显式发送 `@session stop`
- 达到配置的硬上限

停止后必须执行：

- 终止子进程
- 回收 reader/writer 任务
- 移除 `chat_key -> session` 映射
- 清理该会话的临时状态目录

### Hard Limit

硬上限必须做成配置项，例如：

- `system.interactive_session_max_age_seconds`

默认值建议：

- `7200`

回收策略：

- manager 在每次 `send()` 前检查会话年龄
- 超限则先关闭会话，再返回“会话已超时关闭，请重新启动”

## Error Handling

需要明确处理以下场景：

- backend 启动失败
- backend 子进程异常退出
- backend 长时间无响应
- 用户在无活动会话时发送 `@session stop`
- 用户在无活动会话时发送 Claude 原生命令
- 用户在活动会话中发送 `@session start <other backend>`

返回原则：

- 控制命令错误返回清晰、可操作的提示
- backend 退出要带“当前会话已关闭”的明确说明
- 不要把底层实现细节原样泄漏给用户，日志里保留完整错误即可

## Configuration

建议新增配置结构：

```json
{
  "system": {
    "interactive_session_max_age_seconds": 7200
  }
}
```

后续如需要可扩展：

- `interactive_session_idle_timeout_seconds`
- `interactive_backend_default`
- `interactive_backend_workdir`

但首版坚持 YAGNI，只做硬上限。

## Testing Strategy

### Unit Tests

新增 manager 级测试，覆盖：

- 启动新会话
- 复用已有会话
- 显式停止
- 超时回收
- 状态查询
- 并发下同一 `chat_key` 只创建一个会话

### Routing Tests

新增 orchestrator 路由测试，覆盖：

- `@session start claude`
- `@session status`
- `@session stop`
- 会话中普通文本走 backend
- 非会话消息仍走原有 DAG 处理链

### Backend Tests

Claude backend 先以 fake process 或 mock transport 做稳定单元测试，覆盖：

- 启动与关闭
- 发送消息
- 读取回复
- 子进程退出处理
- 状态目录隔离

真实 Claude CLI 集成测试不进入默认 `pytest -q` 主路径，应做成显式 opt-in，以避免认证、网络、权限带来的环境噪音。

## Non-Goals and Follow-Ups

首版完成后，再评估以下增强：

- `CodexInteractiveBackend`
- 会话事件流式转发到飞书卡片
- 会话闲置超时
- 会话 transcript 持久化
- backend 能力发现与动态帮助信息

## Acceptance Criteria

满足以下条件视为设计完成：

- 用户可以通过 `@session start claude` 为当前聊天启动独立 Claude 会话
- 会话只在用户显式启动后创建，不会随 BabyBot 启动自动拉起
- 会话中后续消息直接进入 Claude 长驻进程
- Claude 原生命令可透传
- 用户可以通过 `@session stop` 明确关闭会话
- 会话存在可配置硬上限，超限自动关闭
- 路由与生命周期实现对未来 `Codex` backend 可复用
