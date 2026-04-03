---
name: mcp-server-standards
layer: org
description: MCP Server Tool 设计 10 条准则，适用于所有 MCP 项目
status: active
last-updated: 2026-04-04
---

# @org/mcp-server-standards

> 基于麦当劳中国 MCP、高德地图 MCP 的实践分析。MCP Tool 的消费者是 LLM，不是前端代码，这一根本差异决定了所有设计取舍。

## 核心认知

| 维度 | REST API | MCP Tool |
|------|----------|----------|
| 消费者 | 前端代码（确定性） | LLM（概率性推理） |
| 路由 | URL path + method | LLM 根据 description **自主选择** |
| 参数来源 | 前端表单 / 代码硬编码 | LLM 从**对话上下文**提取 |
| 响应消费 | 代码解析 JSON | LLM 需要**理解并转述** |

## 10 条准则

### C1 · Description 三段式

Tool description 必须包含：① 做什么 ② 什么时候用 ③ 触发短语（可选但推荐）。

```
✅ "获取营养数据。当用户咨询热量时使用。当用户说'多少卡'时触发"
❌ "Query member info"
```

### C2 · 极简参数

能从 token / 上下文推断的参数必须移除。零参数是理想状态。每多一个参数，LLM 填错概率增加一分。

```
✅ ToC 工具无 member_id，从 token 推断
❌ 同时要求传 user_id 和 phone
```

### C3 · 参数扁平化

尽量用 string 等扁平类型。LLM 构造嵌套 JSON 容易出错。

```
✅ location: str  # "116.48,39.99"
❌ location: dict  # {"lng": 116.48, "lat": 39.99}
```

### C4 · 参数来源标注

参数 description 必须说明值从哪个 tool 获取。

```
✅ store_id: str  # 从 nearby_stores 返回结果中获取
❌ store_id: str  # 门店编码
```

### C5 · 命名规范

- 加命名空间前缀（防多 Server 冲突）
- 风格一致（统一 snake_case 或 kebab-case）
- 动词优先（create_order > order_creation）

### C6 · 响应精简

返回字段裁剪到 LLM 所需最小集。每多一个无用字段，多消耗 token 和注意力。

```
✅ POI 搜索只返回 id, name, address, type
❌ 透传后端 20+ 字段
```

### C7 · 响应格式一致

所有 tool 错误格式统一。单位统一（不混用"分"和"元"）。

```
✅ 统一 {"error": "message"}
❌ 有的返回 error，有的返回 msg
```

### C8 · 渐进式披露

List → Detail 模式。列表工具只返回摘要 + ID，详情按需查。

```
✅ browse_menu → id + name + price → drink_detail → 完整信息
❌ 列表返回每条记录全部字段
```

### C9 · 写操作安全

写操作前必须有预览/确认步骤。写操作必须幂等。

```
✅ calculate_price → 用户确认 → create_order
❌ 一个 tool 同时算价和下单
```

### C10 · 敏感信息

响应中 PII 必须脱敏。

```
✅ 152****6666
❌ 15266666666
```

## 设计 Checklist

新建 MCP Tool 前逐条检查：

- [ ] Description 写了"做什么"+"什么时候用"+"用户怎么说"？
- [ ] 参数能从 token/上下文推断的都去掉了？
- [ ] 参数描述标注了"值从哪个 tool 获取"？
- [ ] 响应裁剪到只剩 LLM 需要的字段？
- [ ] 写操作前有确认/预览步骤？
- [ ] Tool name 加了命名空间前缀？
- [ ] 错误格式和其他工具一致？
- [ ] 敏感信息做了脱敏？
