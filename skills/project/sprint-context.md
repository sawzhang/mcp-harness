---
name: sprint-context
layer: project
description: 当前 Sprint 上下文，焦点 FID 和关键决策
status: active
last-updated: 2026-04-04
---

# Sprint Context

Sprint: S-2026-01 (Phase 0)
Deadline: 2026-04-30
焦点: FID-001（语音追加商品）

## 当前目标

1. CLAUDE.md + 核心 Skill 入 repo ✅
2. Schema Lint (C1-C10) 接入 CI
3. Eval Harness MVP（Qianwen × P0 Case）
4. 第一份 Eval Report 产出
5. 从 Eval 失败反推 Tool Description 改进

## 架构决策

- ADR-001: Eval Harness 不直接调 HTTP，让 Agent 自主决策 Tool 调用
- ADR-002: Schema Lint 作为 CI 阻断门，Eval 作为发布阻断门
- ADR-003: 先验证 Qianwen 单模型，Phase 1 再加对照组
