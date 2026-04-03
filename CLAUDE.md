# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP Harness is a **Harness Engineering** practice project for MCP Server quality assurance. It does NOT implement the MCP Server itself (that lives in `../coffee-mcp` with 21 consumer-facing tools for coffee ordering). Instead, it builds a three-layer quality system around MCP: static lint → dynamic eval → automated optimization.

The core insight: MCP Tool consumers are LLMs, not frontend code. Tool descriptions are routing rules — whether a model calls the right tool depends entirely on how well the description is written. This project evaluates **Tool Description × Model** combinations, not server correctness.

## Architecture

```
harness/
├── lint/mcp_schema_lint.py      # Layer 0: Static scan (C1-C10 rules, CI gate)
├── eval/                        # Layer 1: Dynamic evaluation
│   ├── harness.py               #   Multi-model parallel eval engine
│   ├── adapters/                #   Model adapters (Claude tool_use, OpenAI tools format)
│   │   ├── base.py              #     AgentAdapter interface + system prompt
│   │   ├── claude_adapter.py    #     Claude (Anthropic SDK)
│   │   └── openai_adapter.py    #     GPT-4o / Qianwen / Doubao (OpenAI-compatible)
│   ├── case_loader.py           #   YAML eval case loading + filtering
│   └── report.py                #   Report generation (YAML + Rich terminal)
└── optimizer/                   # Layer 2: Automated description optimization
    └── loop.py                  #   autoresearch-style greedy hill-climbing

evals/cases/                     # Eval cases organized by layer
├── tool-selection/              #   Does the model pick the right tool?
├── param-mapping/               #   Does it map "中杯" → tall (not medium)?
├── multi-turn/                  #   Multi-turn context preservation
└── safety/                      #   L3 write safety, injection, jailbreak

skills/                          # Three-layer Skill OS
├── org/mcp-server-standards.md  #   C1-C10 design rules (cross-project)
├── team/coffee-order-domain.md  #   Starbucks domain knowledge
└── project/sprint-context.md    #   Current sprint focus
```

### Three-Layer Pipeline

| Layer | Module | Trigger | Gate |
|-------|--------|---------|------|
| **Lint** (static) | `harness/lint/mcp_schema_lint.py` | Every CI run | C1-C10 errors → PR blocked |
| **Eval** (dynamic) | `harness/eval/harness.py` | PR merge / Sprint | P0 pass rate < 95% → release blocked |
| **Optimize** (iterative) | `harness/optimizer/loop.py` | On demand | Greedy hill-climbing on pass rate |

### Adapter Pattern

Each LLM has a different function-calling format. Adapters convert MCP tool specs to model-native format:
- **ClaudeAdapter**: Anthropic `tool_use` blocks
- **OpenAIAdapter**: OpenAI `tools` / `functions` (also works for Qianwen, Doubao via `base_url`)

The eval tests **the same Tool Description against different models** — format conversion differences are themselves a test surface.

### Optimizer Loop (autoresearch pattern)

Borrowed from [autoresearch](../autoresearch): fixed evaluation + greedy hill-climbing.

```
Round 0: Baseline eval → record pass_rate
Round N: LLM analyzes failures → proposes description changes → re-eval
         → pass_rate improved? KEEP : REVERT
         → 3 rounds without improvement? STOP
```

Output: `evals/optimized/optimized_tools.yaml` — best Tool descriptions found.

## Development Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_schema_lint.py -v

# Run a single test
uv run pytest tests/test_eval_harness.py::test_tool_selection_pass -v

# Lint coffee-mcp server (scans @mcp.tool() decorators, C1-C10)
uv run python -m harness.lint.mcp_schema_lint ../coffee-mcp/src/coffee_mcp/toc_server.py

# List eval cases without running
uv run python scripts/run_eval.py --dry-run --cases evals/cases/

# Run eval (needs API key)
DASHSCOPE_API_KEY=xxx uv run python scripts/run_eval.py --model qianwen --cases evals/cases/

# Run optimizer (mock mode, no API key needed)
uv run python scripts/run_optimize.py --mock --rounds 5

# Run optimizer with real models
DASHSCOPE_API_KEY=xxx ANTHROPIC_API_KEY=xxx \
uv run python scripts/run_optimize.py --model qianwen --rounds 10
```

## Eval Case YAML Structure

```yaml
layer: tool_selection          # tool_selection | param_mapping | multi_turn | safety
criticality: P0                # P0 (blocks release) | P1 (track) | P2 (informational)
cases:
  - id: TS-001
    user_instruction: "再来一杯冰美式大杯"
    expected_tool: calculate_price
    expected_not_tool: create_order      # should NOT be called
    expected_params:                      # partial match — extra fields OK
      items: [{size: "grande"}]
    trap: "模型不应直接调用 create_order"  # documents expected failure mode
    order_context:                        # pre-existing state
      store_id: "ST_SH_001"
```

Multi-turn cases use `dialogue` with per-turn expectations:
```yaml
layer: multi_turn
cases:
  - id: MT-001
    dialogue:
      - turn: 1
        user: "来杯拿铁"
        expected_tool: calculate_price
      - turn: 2
        user: "冰的"
        expected_tool: calculate_price
        expected_params: {items: [{temperature: "iced"}]}
```

Parameter matching uses **partial match**: `expected ⊆ actual`. If the case expects `{size: "tall"}` and the model returns `{size: "tall", quantity: 1, product_code: "D001"}`, it passes.

## Schema Linter Rules (C1-C10)

The linter (`mcp_schema_lint.py`) extracts `@mcp.tool()` decorated functions via AST and checks:

| Rule | What | Severity |
|------|------|----------|
| C1 | Description has ≥2 segments (what + when to use) | WARNING |
| C2 | No params inferable from token (user_id, member_id) | WARNING |
| C3 | No nested dict/object params (LLM can't construct reliably) | WARNING |
| C4 | Reference params (_id, _code) say which tool provides the value | WARNING |
| C5 | Naming: consistent case, no mixed snake/kebab | ERROR |
| C6 | Reference params have example values | INFO |
| C7 | Write ops mention error scenarios in description | INFO |
| C8 | List tools mention returning summary + ID (progressive disclosure) | INFO |
| C9 | Write ops have idempotency_key; L3 ops have confirmation_token | ERROR |
| C10 | PII fields mention masking in description | WARNING |

Exit code 1 (CI gate blocked) if any ERROR-severity issues found.

## Architecture Constraints (Immutable)

1. **Tool Description is the routing rule** — MCP consumers are LLMs; description quality determines call accuracy
2. **Stateless** — each Tool call is independent, no server-side session
3. **Price sovereignty** — server computes prices; client-submitted prices must be ignored
4. **Idempotency** — all L2/L3 write ops require `idempotency_key`, server-side enforced
5. **Confirmation gate** — L3 ops (create_order, stars_redeem) require `confirmation_token` from a prior `calculate_price` call

## Domain Knowledge

- Starbucks size mapping (critical — most common model failure):
  - 中杯 = `tall` (NOT medium), 大杯 = `grande` (NOT large), 超大杯 = `venti`
- Order flow: `nearby_stores → browse_menu → drink_detail → calculate_price → [user confirms] → create_order`
- Security tiers: L0 (public read, 60/min) → L1 (auth read, 30/min) → L2 (write, 5/hr) → L3 (transaction, 10/day)
- The target MCP Server is at `../coffee-mcp` (21 tools in `toc_server.py`, mock data in `toc_mock_data.py`)

## Skills

- `skills/org/mcp-server-standards.md` — C1-C10 design rules, applicable to all MCP projects
- `skills/team/coffee-order-domain.md` — Starbucks modifier mappings, order flow, known model-specific issues
- `skills/project/sprint-context.md` — Current sprint focus and ADRs
