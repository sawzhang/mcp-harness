---
name: mcp-quality
description: MCP Server Tool 开发质量保障。当开发、修改、审查 MCP Tool 定义时使用。当用户说"review MCP tools"、"检查工具设计"、"跑一下 eval"时触发。
---

# MCP Tool 开发质量保障

你正在开发或修改 MCP Server 的 Tool 定义。MCP Tool 的消费者是 LLM，不是前端代码。Tool Description 写得好不好，直接决定不同模型能否正确调用。

## 实现阶段：写 Tool 时的检查清单

每写完一个 `@mcp.tool()` 函数，立即自检：

### C1 · Description 三段式
```
第一句：做什么
第二句：什么场景下用
第三句（推荐）：用户可能怎么说
```

### C2 · 极简参数
能从 token/上下文推断的参数必须删除。零参数是理想状态。

### C3 · 参数扁平化
避免 dict/object 嵌套。LLM 构造嵌套 JSON 容易出错。用 string 替代。

### C4 · 参数来源标注
引用型参数（_id, _code）必须在 description 中写明"从哪个 tool 获取"。
```python
# ✅
store_id: str  # 门店ID，从 nearby_stores 返回结果中获取
# ❌
store_id: str  # 门店编码
```

### C9 · 写操作安全
- 所有写操作必须有 `idempotency_key` 参数
- L3 高风险操作（下单、兑换）必须有 `confirmation_token`
- 写操作前必须有预览/确认步骤（如 calculate_price → create_order）

### 杯型映射（星巴克场景关键）
size description 必须显式写明：`tall(中杯12oz) | grande(大杯16oz) | venti(超大杯20oz)`
不写的后果：Qianwen 会把"中杯"映射为 medium，GPT 会映射为 small。

## 测试阶段：验证 Tool 质量

### Step 1: 静态扫描（秒级）
```bash
uv run python -m harness.lint.mcp_schema_lint <server.py>
```
- ERROR（C5 命名、C9 幂等） → 必须修复
- WARNING（C1-C4, C10） → 应该修复
- INFO（C6-C8） → 建议优化

### Step 2: 多模型评估（分钟级）
```bash
uv run python scripts/run_eval.py --model qianwen --cases evals/cases/
```
验证 Qianwen/Claude/GPT 能否根据 Description 正确选工具、正确构造参数。

P0 通过率 < 95% → 阻断发布。根因通常是 Description 不够明确。

### Step 3: 自动优化（可选）
```bash
uv run python scripts/run_optimize.py --model qianwen --rounds 10
```
LLM 自动分析失败、修改 Description、重跑评估、保留改进、回滚退步。

## 审查阶段：Review MCP Tool PR 时的重点

1. **Description 能让不同模型正确路由吗？** — 不只看人能不能读懂
2. **参数来源链路完整吗？** — 每个引用型参数能追溯到来源 Tool
3. **写操作有确认前置吗？** — calculate_price → create_order，不能跳步
4. **PII 脱敏了吗？** — 手机号 152****6666，不能返回全量
5. **Eval 报告通过了吗？** — P0 通过率 ≥ 95%

## 已知的模型特异性问题

| 问题 | 模型 | 根因 | 解法 |
|------|------|------|------|
| "中杯"→medium | Qianwen | Description 只写 tall 没写中文 | 加 `tall(中杯12oz)` |
| 编造不存在的参数 | GPT-4o | Description 没说"无需传参" | 加 `无需指定xxx` |
| 多轮遗忘 modifier | Claude | 上下文窗口自然衰减 | 关键信息重复确认 |
| 跳过确认直接下单 | 所有 | create_order description 没强调前置 | 加 `必须先调用 calculate_price` |
