"""市场中性 Top/Bottom 策略。

继承 qlib.contrib.strategy.signal_strategy.WeightStrategyBase，
在月末调仓日做多 topk 只最高分股票、做空 bottomk 只最低分股票。
非调仓日返回 None 以保持持仓漂移。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.data import D
from qlib.log import get_module_logger

logger = get_module_logger(__name__)


class TopBottomNeutralStrategy(WeightStrategyBase):
    """月末调仓的市场中性 Top/Bottom 策略。

    在每月最后一个交易日：
    - 按 score 降序排列
    - 前 topk 只股票各分配 +1/topk 权重（做多）
    - 后 bottomk 只股票各分配 -1/bottomk 权重（做空）
    - 其余股票权重为 0

    非月末日返回 None，持仓随市场价格漂移。

    Parameters
    ----------
    topk : int
        做多股票数量，默认 50。
    bottomk : int
        做空股票数量，默认 50。
    rebalance_freq : str
        调仓频率，默认 "month_end"（月末）。
    """

    def __init__(
        self,
        *,
        topk: int = 50,
        bottomk: int = 50,
        rebalance_freq: str = "month_end",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topk = topk
        self.bottomk = bottomk
        self.rebalance_freq = rebalance_freq
        self._last_target_weights: Optional[dict] = None

        logger.info(
            "TopBottomNeutralStrategy initialized: topk=%d, bottomk=%d, freq=%s",
            topk, bottomk, rebalance_freq,
        )

    def generate_target_weight_position(
        self,
        score: pd.Series,
        current,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> Optional[dict]:
        """生成目标持仓权重。

        Parameters
        ----------
        score : pd.Series
            当前可交易股票的预测分数，index 为 instrument。
        current : Position
            当前持仓。
        trade_start_time : pd.Timestamp
            调仓起始时间。
        trade_end_time : pd.Timestamp
            调仓结束时间。

        Returns
        -------
        dict or None
            {instrument: weight} 或 None（表示保持当前持仓不变）。
        """
        if not self._is_rebalance_day(trade_start_time):
            return None

        # 剔除 NaN score
        valid_scores = score.dropna()
        if len(valid_scores) < max(self.topk, self.bottomk):
            logger.warning(
                "insufficient valid scores on %s: %d < %d, skipping rebalance",
                trade_start_time.date(), len(valid_scores), max(self.topk, self.bottomk),
            )
            return None

        sorted_scores = valid_scores.sort_values(ascending=False)

        long_stocks = sorted_scores.head(self.topk).index.tolist()
        short_stocks = sorted_scores.tail(self.bottomk).index.tolist()

        # 去重保护：如果 N < topk + bottomk，去重后可能少于预期
        long_set = set(long_stocks)
        short_stocks = [s for s in short_stocks if s not in long_set]

        weights: dict = {}
        for stock in long_stocks:
            weights[stock] = 1.0 / self.topk
        for stock in short_stocks:
            weights[stock] = -1.0 / self.bottomk

        logger.debug(
            "rebalance %s: long=%d, short=%d, total_weight=%.6f, abs_sum=%.6f",
            trade_start_time.date(),
            len(long_stocks), len(short_stocks),
            sum(weights.values()),
            sum(abs(w) for w in weights.values()),
        )

        self._last_target_weights = weights
        return weights

    def _is_rebalance_day(self, trade_start_time: pd.Timestamp) -> bool:
        """使用 Qlib Calendar API 判定是否为月末交易日。

        Parameters
        ----------
        trade_start_time : pd.Timestamp
            当前交易日。

        Returns
        -------
        bool
        """
        if self.rebalance_freq == "month_end":
            # 获取该月最后一个交易日
            year = trade_start_time.year
            month = trade_start_time.month
            month_start = pd.Timestamp(year=year, month=month, day=1)
            # 获取下个月第一天
            if month == 12:
                next_month_start = pd.Timestamp(year=year + 1, month=1, day=1)
            else:
                next_month_start = pd.Timestamp(year=year, month=month + 1, day=1)

            try:
                # 获取当月所有交易日，最后一天即为月末交易日
                calendar = D.calendar(
                    start_time=month_start,
                    end_time=next_month_start,
                    freq="day",
                )
                if len(calendar) > 0:
                    last_trading_day = calendar[-1]
                    return trade_start_time == last_trading_day
            except Exception:
                pass
            return False

        # 其他频率暂不支持
        logger.warning("unsupported rebalance_freq: %s", self.rebalance_freq)
        return False
