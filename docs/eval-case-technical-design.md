# Eval Case 技术方案

> MCP Harness Eval 层的核心技术设计文档。描述 Eval Case 的数据模型、分类体系、匹配算法、执行引擎、报告生成以及与 Optimizer 的协作机制。

## 1. 设计目标

MCP Harness 不测 Server 正确性，测的是 **Tool Description × Model** 的组合效果。Eval Case 回答一个核心问题：

> 给定一段 Tool Description，模型 X 能否根据用户的自然语言指令，选对工具、构造对参数？

每个 Eval Case 是一个 **断言（assertion）**：用户说了什么 → 期望模型调用什么工具、传入什么参数。整个 Eval 体系的技术挑战在于：

- **多模型并行**：同一组 Case 要在 Claude / Qianwen / GPT-4o 上同时运行，格式转换差异本身就是测试面
- **部分匹配**：模型返回的参数可能比期望多（extra fields OK），需要 `expected ⊆ actual` 的递归匹配
- **多轮状态**：多轮对话中每一轮各自有期望，且需要追踪上下文累积
- **异构期望**：有些 Case 期望调用某工具（正向），有些期望不调用某工具（负向），有些只描述行为（语义断言）

## 2. Case 数据模型

### 2.1 EvalCase 结构

```python
@dataclass
class EvalCase:
    # === 标识 ===
    id: str                    # 全局唯一 ID（如 TS-001, PE-COMBO-002）
    layer: str                 # 所属一级分类（tool_selection, param_mapping, ...）
    criticality: str           # P0 (blocks release) | P1 (track) | P2 (informational)
    tags: list[str]            # 用于过滤的标签
    source: str                # 来源: manual | fid_generated | prod_distilled | adversarial

    # === 单轮输入 ===
    user_instruction: str      # 用户自然语言指令
    order_context: dict        # 预置上下文（store_id, product_code 等）

    # === 断言 ===
    expected_tool: str         # 期望调用的工具名
    expected_not_tool: str     # 期望不调用的工具名（负向断言）
    expected_params: dict      # 期望参数（部分匹配）
    expected_params_semantic: str  # 语义级参数断言（自然语言描述）
    expected_behavior: str     # 行为级断言（如"模型应提醒用户"）
    violation: str             # 违规描述
    trap: str                  # 该 Case 的陷阱说明（文档用途）

    # === 多轮 ===
    dialogue_turns: list[dict] # 多轮对话，每轮含 user + 各自的期望

    # === 参数映射专用 ===
    mapping_tests: list[dict]  # 同一 Case 的多组映射变体
```

### 2.2 YAML 序列化格式

**单轮 Case：**

```yaml
layer: tool_selection
criticality: P0
tags: [tool_selection, ambiguous]
source: manual

cases:
  - id: TS-AMB-001
    user_instruction: "有什么优惠活动"
    expected_tool: campaign_calendar
    expected_not_tool: available_coupons
    trap: "优惠活动是活动日历，不是可领优惠券"
    order_context:
      store_id: "ST_SH_001"
```

**多轮 Case：**

```yaml
layer: multi_turn
criticality: P0

cases:
  - id: MT-MOD-001
    dialogue:
      - turn: 1
        user: "来杯大杯冰拿铁"
        expected_tool: calculate_price
        expected_params:
          items:
            - product_code: "D003"
              size: "grande"
      - turn: 2
        user: "还是要热的吧"
        expected_tool: calculate_price
        expected_params:
          items:
            - product_code: "D003"
              size: "grande"        # 杯型必须保持不变
        trap: "改温度但杯型 grande 必须保持"
```

**设计约定：**
- 文件级 `layer` / `criticality` / `tags` 作为默认值，Case 级可覆盖
- `order_context` 注入到 system prompt 或对话前置上下文中，模拟真实场景的前序状态
- `trap` 字段是人类可读的陷阱说明，不参与自动匹配，用于报告和 Optimizer 分析

## 3. Case 分类体系

采用两级分类：一级分类对应评估维度，二级分类对应具体测试场景。

### 3.1 分类全景

```
evals/cases/
├── tool-selection/          一级: 工具路由
│   ├── ts-basic.yaml          二级: 基础路由（单工具选择）
│   ├── ts-ambiguous.yaml      二级: 歧义消歧（相似工具间区分）
│   ├── ts-chain.yaml          二级: 链式依赖（前置工具感知）
│   └── ts-coverage.yaml       二级: 工具全覆盖（21 工具逐个验证）
│
├── param-mapping/           一级: 参数映射（原有）
│   └── pm-size-mapping.yaml   二级: 杯型映射
│
├── param-extraction/        一级: 参数提取
│   ├── pe-combo.yaml          二级: 多维组合（一句话多参数）
│   ├── pe-alias.yaml          二级: 别名/口语化（澳白、续命水）
│   ├── pe-modifier.yaml       二级: 单维修饰（甜度、奶型扩展）
│   ├── pe-quantity.yaml       二级: 数量提取（中文数词解析）
│   └── pe-disambiguation.yaml 二级: 产品消歧（模糊品名匹配）
│
├── customization/           一级: 商品客制化
│   ├── cx-constraint.yaml     二级: 产品约束（星冰乐只能冰）
│   ├── cx-availability.yaml   二级: 规格可用性（杯型/奶型限制）
│   ├── cx-default.yaml        二级: 默认值推断（只说品名时）
│   └── cx-combo-constraint.yaml 二级: 组合约束（互斥选项）
│
├── multi-turn/              一级: 多轮对话
│   ├── mt-incremental-order.yaml  二级: 上下文保持（原有）
│   ├── mt-modify-rollback.yaml    二级: 修改与回退
│   ├── mt-multi-item.yaml         二级: 多品并行点单
│   ├── mt-cancel-restart.yaml     二级: 取消与重来
│   └── mt-long-context.yaml       二级: 长对话衰减（5+ 轮）
│
├── workflow/                一级: 流程编排
│   ├── wf-prerequisite.yaml   二级: 前置依赖（不跳步）
│   ├── wf-pickup.yaml         二级: 取餐方式（自提/外送/堂食）
│   ├── wf-coupon.yaml         二级: 优惠券流程
│   └── wf-reorder.yaml        二级: 复购流程
│
└── safety/                  一级: 安全防护
    ├── sf-order-safety.yaml   二级: 确认机制 + 注入防护（原有）
    ├── sf-boundary.yaml       二级: 边界值（极端参数）
    ├── sf-auth-ext.yaml       二级: 权限控制扩展
    ├── sf-duplicate.yaml      二级: 重复操作防护
    └── sf-info-leak.yaml      二级: 信息泄露防护
```

### 3.2 分类定义

| 一级分类 | 评估维度 | Case 数 | 测什么 |
|---------|---------|--------|-------|
| **tool-selection** | 模型能否选对工具 | 23 | 给定用户意图，模型是否调用了正确的 MCP 工具 |
| **param-extraction** | 模型能否提取对参数 | 19 | 从自然语言中提取多维参数并正确映射到 API 字段 |
| **param-mapping** | 领域参数映射 | 7 | 星巴克特定映射（中杯=tall 不是 medium） |
| **customization** | 商品客制化理解 | 15 | 产品约束规则、规格限制、默认值、互斥选项 |
| **multi-turn** | 多轮上下文能力 | 10 | 参数累积、修改回退、多品管理、长对话衰减 |
| **workflow** | 流程编排能力 | 10 | 工具调用顺序、前置依赖、取餐/优惠券/复购流程 |
| **safety** | 安全与边界 | 17 | 确认跳过、注入、越权、边界值、重复操作、信息泄露 |

### 3.3 分类设计原则

**为什么不用单一维度（如 "tool-selection" + "param" 二分法）？**

因为真实用户的失败模式是多维的。以「馥芮白换燕麦奶」为例：

- 从 tool-selection 看，模型选对了 `calculate_price` ✓
- 从 param-extraction 看，模型提取了 `milk: "oat"` ✓
- 从 customization 看，馥芮白不支持换奶 ✗

单一维度的分类会让这类失败淹没在 "param" 大类里。独立出 `customization` 层后，可以精确定位"模型理解了参数但不理解业务约束"这类问题。

**分类与目录的映射：**

一级分类 = 目录名。二级分类 = 文件名前缀（`ts-`, `pe-`, `cx-`, `mt-`, `wf-`, `sf-`）。Case ID 前缀与二级分类对应，保证全局唯一且可追溯：

```
TS-001       → tool-selection / ts-basic
PE-COMBO-002 → param-extraction / pe-combo
CX-CONST-003 → customization / cx-constraint
MT-MOD-001   → multi-turn / mt-modify-rollback
WF-PICK-002  → workflow / wf-pickup
SF-BOUND-001 → safety / sf-boundary
```

## 4. 匹配算法

### 4.1 部分匹配（Partial Match）

核心规则：`expected ⊆ actual`。期望中指定的字段必须出现在实际返回中，实际返回可以有额外字段。

```python
def _param_match(expected, actual) -> bool:
    if expected == actual:
        return True
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            k in actual and _param_match(v, actual[k])
            for k, v in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_param_match(e, a) for e, a in zip(expected, actual))
    return expected == actual
```

**示例：**

```yaml
# Case 期望
expected_params:
  items:
    - size: "grande"

# 模型实际返回
items:
  - product_code: "D003"
    size: "grande"
    quantity: 1
    milk: "whole"

# 匹配结果: PASS（extra fields OK）
```

**设计权衡：**

- 列表匹配采用**顺序敏感**策略（`zip`），因为 items 顺序在咖啡点单中有业务含义（先点的先做）
- 如果未来需要顺序无关匹配，可扩展为集合匹配模式（需在 Case 中标注 `match_mode: unordered`）

### 4.2 三层断言

每个 Case 最多包含三层断言，按优先级执行：

```
Layer 1: Tool Selection
  ├─ expected_tool      → 正向：是否调用了期望工具
  └─ expected_not_tool  → 负向：是否避免了危险工具

Layer 2: Parameter Match
  └─ expected_params    → 部分匹配：期望参数 ⊆ 实际参数

Layer 3: Behavior Assertion（当前为文档记录，待实现自动化）
  ├─ expected_behavior  → 语义断言（如"模型应提醒用户"）
  └─ trap               → 陷阱说明（人工审查用）
```

**Case 通过条件：** `所有 checks 均 passed`。任一 check 失败即 Case 失败。

### 4.3 多轮匹配

多轮 Case 的每一轮独立评估，但共享对话上下文：

```
Turn 1: user → adapter.run_dialogue() → checks[turn_1_tool, turn_1_params]
         ↓ conversation_history
Turn 2: user + history → adapter.run_dialogue() → checks[turn_2_tool, turn_2_params]
         ↓ conversation_history
Turn 3: user + history → adapter.run_dialogue() → checks[turn_3_tool, turn_3_params]
```

**上下文传递机制：**

Adapter 的 `run_multi_turn()` 将历史对话以 `[对话历史]` 前缀拼入下一轮 `user_message`，让单轮 Adapter 也能感知上下文。历史中包含 tool call 摘要：

```
[对话历史]
用户: 来杯大杯冰拿铁
助手: 好的 [调用了: calculate_price({"items": [{"product_code": "D003", "size": "grande"}]})]

[当前用户消息]
还是要热的吧
```

**多轮通过条件：** 所有轮次的所有 checks 均 passed。

## 5. 执行引擎

### 5.1 架构

```
                          ┌─────────────────┐
                          │  MCPEvalHarness  │
                          │                  │
                          │  mcp_tools[]     │ ← Tool Description (评估对象)
                          │  models{}        │ ← 已注册的 Model Adapters
                          └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │  Claude   │  │  Qianwen │  │  GPT-4o  │
            │  Adapter  │  │  Adapter  │  │  Adapter  │
            └──────────┘  └──────────┘  └──────────┘
                    │              │              │
                    ▼              ▼              ▼
               tool_use       functions        tools
                format         format          format

               同一份 Tool Description → 不同的传入格式
               格式转换差异本身就是测试面
```

### 5.2 并发控制

```python
async def run_suite(cases, tier_filter=1, concurrency=5):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for model_name, entry in models.items():
        if entry.tier <= tier_filter:
            for case in cases:
                tasks.append(_eval_with_semaphore(semaphore, model_name, entry, case))
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

- **Semaphore 控制并发度**（默认 5），避免 API rate limit
- **Model × Case 笛卡尔积**：N 个模型 × M 个 Case = N×M 个并发任务
- **异常隔离**：单个 Case 异常不影响其他 Case，异常被捕获为 `EvalResult(passed=False, error=...)`

### 5.3 Tier 分层

```
Tier 1: 每次 PR → 只跑 P0 Case + 默认模型    （秒级，CI 可用）
Tier 2: 每个 Sprint → P0+P1 Case + 全模型      （分钟级）
Tier 3: 月度 / 发布前 → 全量 Case + 全模型       （完整基线）
```

通过 `register_model(name, adapter, tier=N)` 控制模型在哪个 Tier 参与评估。

### 5.4 Mock Tool Result

当前实现中，Adapter 对 tool call 返回固定 mock 结果 `{"status": "success", "mock": true}`。这意味着：

- **可以测试**：工具选择、参数构造
- **无法测试**：模型对 tool 返回值的理解和引用

后续可扩展为带结构的 mock result（按 tool name 返回预定义的真实结构），以测试模型对返回值的处理能力。

## 6. 报告与 Gate

### 6.1 报告结构

```yaml
meta:
  generated_at: "2026-04-07T10:30:00"
  total_cases: 303        # 101 cases × 3 models
  models_tested: [claude, qianwen, gpt4o]

pass_rate:
  claude: 92.1%
  qianwen: 78.2%
  gpt4o: 85.1%

by_layer:
  tool_selection:
    claude: 95.7%
    qianwen: 82.6%
  param_mapping:
    claude: 88.9%
    qianwen: 71.4%         # ← 杯型映射失败集中在此
  customization:
    claude: 93.3%
    qianwen: 73.3%
  # ...

top_failure_patterns:
  - pattern: "param_items"    count: 18
  - pattern: "tool_selection" count: 12

gate_decision:
  result: BLOCK
  reason: "qianwen 通过率 78.2% < 95%"
```

### 6.2 Gate 规则

```
IF any(model.pass_rate < 95%) THEN BLOCK
```

Gate 是二值的：PASS 或 BLOCK。被 BLOCK 意味着该模型的 Tool Description 需要进入 Optimizer 优化循环。

### 6.3 报告维度

报告从三个维度交叉呈现结果：

- **Model 维度**：哪个模型整体表现最差？→ 决定优先优化哪个模型的兼容性
- **Layer 维度**：哪类能力缺陷最多？→ 决定 Description 改进方向（补路由信息 vs 补参数说明）
- **Case 维度**：哪些具体 Case 反复失败？→ 输入 Optimizer 的 failure 分析

## 7. 与 Optimizer 的协作

Eval Case 是 Optimizer 的「不可变基础设施」——Optimizer 修改 Tool Description，但永远不修改 Case。

### 7.1 协作流程

```
                    ┌──────────────────────────┐
                    │      Eval Case (固定)     │
                    │     101 cases, 6 layers   │
                    └────────────┬─────────────┘
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      │                          │                          │
      ▼                          ▼                          ▼
  Round 0                    Round N                    Round N+1
  Baseline                   Propose                    Accept/Reject
  ┌──────────┐              ┌──────────┐              ┌──────────┐
  │ Eval     │──failures──→ │ Analyzer │──changes──→  │ Eval     │
  │ 58.3%    │              │ (LLM)    │              │ 65.2%    │
  └──────────┘              └──────────┘              └──────────┘
                                                       ↓
                                                  pass_rate 提升?
                                                  ├─ YES → KEEP
                                                  └─ NO  → REVERT
```

### 7.2 Optimizer 如何消费 Case 信息

Optimizer 的 Analyzer Prompt 接收失败 Case 的以下字段：

- `case_id`：定位哪个 Case 失败
- `layer`：判断失败属于哪类能力缺陷
- `checks`：具体哪个断言失败（tool_selection / param_items / ...）
- `tool_calls`：模型实际调用了什么（与期望对比）
- `trap`：陷阱说明，帮助 LLM 理解失败根因

**Analyzer 不直接看 `expected_params`**——它看的是失败的 check 信息和实际 tool_calls，从中推断 Description 哪里写得不够清楚。

### 7.3 收敛条件

- pass_rate ≥ 98%（目标达成）
- 连续 3 轮无改进（陷入局部最优）
- 达到 max_rounds

## 8. Case 编写规范

### 8.1 ID 命名

```
{分类前缀}-{子分类}-{序号}

TS-001          tool-selection / basic / 第1个
PE-COMBO-003    param-extraction / combo / 第3个
CX-CONST-001    customization / constraint / 第1个
MT-MOD-002      multi-turn / modify / 第2个
WF-PICK-001     workflow / pickup / 第1个
SF-BOUND-004    safety / boundary / 第4个
```

### 8.2 Criticality 标准

| 等级 | 标准 | 影响 |
|-----|------|------|
| **P0** | 核心流程失败（选错工具、关键参数错误、安全绕过） | 阻断发布 |
| **P1** | 边缘场景（不常见组合、模糊表达、非核心约束） | 追踪但不阻断 |
| **P2** | 体验优化（更好的默认值、更自然的回复） | 仅记录 |

### 8.3 trap 字段编写

`trap` 是给 Optimizer LLM 和人类审查者看的，描述「这个 Case 为什么容易出错」：

```yaml
# 好的 trap
trap: "中杯在星巴克体系中是 tall，不是 medium。模型受通用语义影响常映射错误"

# 差的 trap
trap: "参数错误"   # 太模糊，Optimizer 无法分析
```

### 8.4 expected_behavior 与自动化

`expected_behavior` 当前是纯文档字段（人工审查用），不参与自动匹配。适用于无法用 tool name / param 精确断言的场景：

```yaml
expected_behavior: "模型应告知用户星冰乐只有冰的选项"
```

未来可通过 LLM-as-Judge 模式实现自动化：将模型的 `final_response` 和 `expected_behavior` 一起提交给 Judge LLM，由 Judge 判断是否满足。

## 9. Case 来源与扩展

### 9.1 四种 Case 来源

| Source | 说明 | 示例 |
|--------|------|------|
| `manual` | 手工编写，覆盖核心路径 | 基础工具选择、安全防护 |
| `fid_generated` | 从 FID（Feature Intent Document）推导 | 客制化约束、参数组合 |
| `prod_distilled` | 从生产日志中提炼的真实失败 | 真实用户的表达方式 |
| `adversarial` | 对抗性生成（红队测试） | 注入、越狱、边界值 |

### 9.2 基于 AIChatOrder 的扩展依据

当前 101 个 Case 中的客制化维度来源于 AIChatOrder 项目的产品模型：

| AIChatOrder 维度 | 对应 Case 分类 | 覆盖度 |
|-----------------|---------------|--------|
| Cupsize（杯型） | pe-size, cx-availability | 已覆盖 tall/grande/venti 映射 + 产品限制 |
| Temperature（温度） | pe-modifier, cx-constraint | 已覆盖热/冰 + 星冰乐只能冰 |
| Milk（奶型） | pe-modifier, cx-constraint | 已覆盖 6 种奶 + 馥芮白不能换奶 |
| ChangeSugar（甜度） | pe-modifier, cx-combo-constraint | 已覆盖标准/半糖/无糖/代糖 + 互斥 |
| EepresMopFilter（浓缩） | pe-combo | 已覆盖加 shot |
| MultiChangeSyrup（风味糖浆） | pe-combo | 已覆盖香草/焦糖/榛果糖浆 |
| FoamMopFilter（奶泡） | cx-combo-constraint | 已覆盖冰饮隐藏奶泡选项 |
| 产品别名 | pe-alias | 已覆盖澳白/续命水/卡布/dirty/抹茶 |
| 产品约束 | cx-constraint | 已覆盖 4 个产品的客制化限制 |

### 9.3 扩展方向

当前 101 个 Case 作为基础评估集，后续可从以下方向扩展：

- **错误恢复**：tool call 返回错误码（售罄、门店关闭）后模型的反应
- **推荐场景**：模型主动推荐相关产品（需要 `browse_menu` → 匹配偏好）
- **多语言**：同一意图的英文 / 中英混杂表达
- **模型特异性**：针对特定模型已知弱点的靶向 Case

## 10. 技术栈依赖

| 组件 | 依赖 | 用途 |
|------|------|------|
| Case 加载 | PyYAML | YAML 解析 |
| 并发执行 | asyncio | 多模型 × 多 Case 并行 |
| Claude Adapter | anthropic SDK | tool_use 格式 |
| OpenAI Adapter | openai SDK | tools/functions 格式 |
| 报告输出 | Rich (optional) | 终端表格渲染 |
| Optimizer | anthropic / openai | LLM 分析失败并提案 |
