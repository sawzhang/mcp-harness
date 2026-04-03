---
name: coffee-order-domain
layer: team
description: 咖啡点单领域知识，星巴克杯型映射、订单流程、modifier 组合规则
status: active
last-updated: 2026-04-04
---

# @team/coffee-order-domain

## 杯型映射（关键！）

星巴克杯型体系与常规认知不同，是所有模型最容易出错的地方：

| 用户说法 | 正确映射 | 常见错误 | 容量 |
|---------|---------|---------|------|
| 中杯 | `tall` | ~~medium~~, ~~small~~ | 12oz / 355ml |
| 大杯 | `grande` | ~~large~~ | 16oz / 473ml |
| 超大杯 | `venti` | ~~extra-large~~, ~~xl~~ | 20oz / 591ml |
| 最大的 | `venti` | — | 语义推理 |

**Tool Description 必须显式写明中文映射**：`size: tall(中杯12oz) | grande(大杯16oz) | venti(超大杯20oz)`

## Modifier 组合规则

### 合法维度

| 维度 | 可选值 | 默认值 |
|------|-------|--------|
| size | tall, grande, venti | grande |
| milk | whole, skim, oat, almond, soy, coconut | whole |
| temperature | hot, iced, blended | hot |
| sweetness | normal, less, half, none | normal |
| extras | extra_shot, vanilla_syrup, caramel_syrup, ... | 无 |

### 非法组合

- blended + 食品类 → 400
- 部分饮品不支持 blended（如美式只有 hot/iced）

### 口语映射

| 用户说法 | 正确映射 |
|---------|---------|
| "去冰" | temperature=iced + extras 不含冰（业务层特殊处理） |
| "常温" | temperature=iced（不加冰，不是 warm） |
| "少糖" | sweetness=less |
| "换燕麦奶" | milk=oat |
| "加浓" / "加一份浓缩" | extras=["extra_shot"] |
| "脱脂" | milk=skim |

## 订单流程

```
nearby_stores(city)          # 1. 找门店
    ↓ store_id
browse_menu(store_id)        # 2. 看菜单（List，返回摘要）
    ↓ product_code
drink_detail(product_code)   # 3. 看详情（Detail，返回定制选项）
    ↓ 用户选择 modifier
calculate_price(store_id, items, coupon_code?)  # 4. 算价（返回 confirmation_token）
    ↓ 用户确认 + confirmation_token
create_order(store_id, items, pickup_type, idempotency_key, confirmation_token)  # 5. 下单
    ↓ order_id
order_status(order_id)       # 6. 查状态
```

**关键约束**：
- Step 4 → Step 5 之间必须有用户确认（Agent 不可自动跳过）
- confirmation_token 有效期有限，过期需重新 calculate_price
- idempotency_key 防止重复下单，24h 内同 key 返回相同结果

## 安全分级

| 级别 | 操作 | 限流 |
|------|------|------|
| L0 | browse_menu, drink_detail, nutrition_info, nearby_stores | 60/min |
| L1 | my_account, my_coupons, calculate_price, order_status | 30/min |
| L2 | claim_all_coupons, create_address | 5/hour |
| L3 | create_order, stars_redeem | 10/day |

## 已知模型特异性

| 模型 | 已知问题 | 应对 |
|------|---------|------|
| Qianwen-Max | "中杯"偶尔映射为 MEDIUM | Description 强化中文标注 |
| GPT-4o | 有时编造不需要的参数（如 member_id） | C2 极简参数 + 参数幻觉测试 |
| Claude | 表现稳定，偶尔在多轮对话中遗忘 modifier | 多轮上下文保持测试 |
