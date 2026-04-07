# MCP Harness

MCP Server 的消费者侧质量保障工具链。

MCP Tool 的消费者是 LLM，不是前端代码。Tool Description 写得好不好，直接决定不同模型能否正确调用。MCP Harness 通过**静态扫描 → 多模型评估 → 自动优化**三层防线，确保 Tool Description 在所有主流模型上都能被正确理解。

可独立使用，也可作为 [superpowers](https://github.com/obra/superpowers) 等 AI 开发工作流的质量 Skill 注入（见 `skills/mcp-quality/SKILL.md`）。

## 三层防线

```
Layer 0 · Schema Lint ──── C1-C10 静态检查（秒级，CI 阻断）
    ↓
Layer 1 · Eval Harness ─── 多模型 × 多场景并行评估（分钟级）
    ↓
Layer 2 · Optimizer ─────── autoresearch 模式自动优化 Description（迭代收敛）
```

## Quick Start

```bash
# 安装
uv sync

# 扫描 MCP Server Tool 定义（C1-C10 规则）
uv run python -m harness.lint.mcp_schema_lint ../coffee-mcp/src/coffee_mcp/toc_server.py

# 列出所有 Eval Case
uv run python scripts/run_eval.py --dry-run

# 运行评估（Mock 模式，无需 API Key）
uv run python scripts/run_optimize.py --mock --rounds 5

# 运行评估（真实模型）
DASHSCOPE_API_KEY=xxx uv run python scripts/run_eval.py --model qianwen

# 自动优化 Tool Description
DASHSCOPE_API_KEY=xxx ANTHROPIC_API_KEY=xxx \
uv run python scripts/run_optimize.py --model qianwen --rounds 10

# 测试
uv run pytest tests/ -v
```

## 核心理念

### 评的不是 Server，是 Description × Model

传统 API 测试验证 Server 逻辑是否正确。MCP Harness 验证的是：**不同模型能否根据 Tool Description 正确选择工具、正确构造参数**。

```
用户："来一杯中杯冰美式"
                ↓
Agent（Qianwen）→ 读 Tool Description → size="medium" ❌  Description 没写清楚
Agent（Claude） → 读 Tool Description → size="tall"   ✅  理解了星巴克映射
```

### Schema Lint (C1-C10)

基于麦当劳中国 MCP、高德地图 MCP 的实践总结的 10 条设计准则：

| 规则 | 检查内容 | 级别 |
|------|---------|------|
| C1 | Description 三段式（做什么 + 什么时候用 + 触发短语） | WARNING |
| C2 | 极简参数（能推断的不要传） | WARNING |
| C3 | 参数扁平化（避免嵌套 dict） | WARNING |
| C4 | 参数来源标注（从哪个 tool 获取） | WARNING |
| C5 | 命名规范（风格一致，动词优先） | ERROR |
| C6 | 参数示例值 | INFO |
| C7 | 写操作提及错误场景 | INFO |
| C8 | 渐进式披露（List → Detail） | INFO |
| C9 | 写操作幂等 + L3 确认前置 | ERROR |
| C10 | PII 脱敏 | WARNING |

### Eval Case 四层覆盖

```
evals/cases/
├── tool-selection/    模型能否选对工具（"帮我领券" → claim_all_coupons）
├── param-mapping/     参数映射（"中杯" → tall，不是 medium）
├── multi-turn/        多轮上下文保持（改温度后杯型不丢）
└── safety/            安全（确认跳过、注入、越狱、价格篡改）
```

### Optimizer（autoresearch 模式）

借鉴 [autoresearch](../autoresearch) 的贪心爬山模式：

```
Round 0: Baseline → 58% pass rate
Round 1: LLM 分析失败 → 改 Description → 重新 Eval → 65% → KEEP
Round 2: 改 Description → 62% → REVERT（回滚）
Round 3: 改 Description → 72% → KEEP
...直到 ≥ 98% 或连续 3 轮无改进
```

产出 `evals/optimized/optimized_tools.yaml`，可直接替换 MCP Server 的 Tool Description。

## 项目结构

```
mcp-harness/
├── CLAUDE.md                    # 项目宪法
├── harness/
│   ├── lint/mcp_schema_lint.py  # C1-C10 静态扫描
│   ├── eval/
│   │   ├── harness.py           # 多模型并行评估引擎
│   │   ├── adapters/            # Claude / OpenAI / Qianwen 适配器
│   │   ├── case_loader.py       # YAML Case 加载
│   │   └── report.py            # 评估报告生成
│   └── optimizer/loop.py        # Description 自动优化循环
├── evals/cases/                 # 24 个评估用例（4 层分类）
├── skills/                      # Skill OS
│   ├── org/                     # 跨项目通用规范
│   ├── team/                    # 领域知识（咖啡点单）
│   ├── project/                 # Sprint 上下文
│   └── mcp-quality/SKILL.md    # 可注入 superpowers 等工作流的质量 Skill
├── fids/                        # Feature Intent Document
├── scripts/                     # CLI 入口
└── tests/                       # 39 个测试
```

## 被评估的 MCP Server

目标 Server 位于 [coffee-mcp](https://github.com/sawzhang/coffee-mcp)，包含 21 个消费者端 Tool（语音点单场景），安全分级 L0-L3。

## License

MIT
