# Layer 4: 策略

## 职责
将预测信号转换为日频目标持仓。月末调仓，做多 topk 最高分、做空 bottomk 最低分，等权重。

## 使用的 Qlib 组件
- `D.calendar` — 月末交易日判定
- （Option C 时使用 `TopBottomNeutralStrategy(WeightStrategyBase)` 自定义类）

## 输入
- `outputs/predictions/predictions_lgbm.csv` — 预测信号
- `configs/strategies/top_bottom_neutral.yaml` — 策略参数（topk, bottomk）
- `configs/_common.yaml` — 全局配置

## 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/positions/positions_lgbm.csv` | datetime, instrument, weight | 日频持仓 |

## 策略规则
- 月末最后一个交易日调仓
- 按 ALPHA 降序排列
- 前 topk 只等权做多（+1/topk）
- 后 bottomk 只等权做空（-1/bottomk）
- 非调仓日延续上期持仓

## 运行
```bash
python 4_strategy/06_strategy_top_bottom/generate_positions.py \
    --predictions outputs/predictions/predictions_lgbm.csv
```
