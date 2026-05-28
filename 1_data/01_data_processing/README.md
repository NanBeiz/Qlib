# Layer 1: 数据准备

## 职责
生成 point-in-time 可交易 CSI500 股票池。

## 使用的 Qlib 组件
- `D.instruments` / `D.list_instruments` — 获取成分股时间区间
- `D.calendar` — 交易日历
- `D.features` — 拉取 $volume、$is_st 过滤字段

## 输入
- `configs/_common.yaml` — 全局配置（market、日期范围）

## 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/data/universe_csi500.csv` | datetime, instrument | 日频可交易股票池 |
| `outputs/data/universe_stats.csv` | datetime, count | 每日成分股数量（审计用） |

## 过滤规则
- `$is_st == 0` — 排除 ST 股票
- `$volume > 0` — 排除停牌/零成交
- 涨跌停不在此层过滤，交给 Layer 5 的 Exchange.limit_threshold

## 运行
```bash
python 1_data/01_data_processing/prepare_data.py --config configs/_common.yaml
```
