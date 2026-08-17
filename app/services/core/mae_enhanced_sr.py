"""
M.A.E. Enhanced Support/Resistance Detector
Adds psychological levels, volume profile nodes, and order blocks
on top of standard swing S/R. Returns list of dicts compatible with V5.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedSRDetector:
    def find_levels(self, candles: List[Any], pair: str = "", cluster_threshold: float = 0.005) -> List[Dict]:
        if len(candles) < 20:
            return []

        all_levels: List[Dict] = []

        # 1. Swing high/low S/R
        swing = self._swing_levels(candles, cluster_threshold)
        all_levels.extend(swing)

        # 2. Psychological round numbers
        psych = self._psychological_levels(candles, pair)
        all_levels.extend(psych)

        # 3. Volume profile high-volume nodes
        if any(getattr(c, "volume", None) for c in candles):
            vp = self._volume_profile_levels(candles, cluster_threshold)
            all_levels.extend(vp)

        # 4. Order blocks
        ob = self._order_block_levels(candles)
        all_levels.extend(ob)

        # Merge & sort by strength
        all_levels = self._merge_levels(all_levels, cluster_threshold)
        all_levels.sort(key=lambda x: x.get("strength", 0), reverse=True)
        return all_levels[:15]

    def _swing_levels(self, candles, cluster_threshold):
        swing_highs = []
        swing_lows = []
        for i in range(2, len(candles) - 2):
            c = candles[i]
            if (c.high > candles[i-1].high and c.high > candles[i-2].high
                    and c.high > candles[i+1].high and c.high > candles[i+2].high):
                swing_highs.append(c.high)
            if (c.low < candles[i-1].low and c.low < candles[i-2].low
                    and c.low < candles[i+1].low and c.low < candles[i+2].low):
                swing_lows.append(c.low)

        res = self._cluster_levels(swing_highs, cluster_threshold)
        sup = self._cluster_levels(swing_lows, cluster_threshold)

        results = []
        for level, count in res:
            touches = self._count_touches(candles, level, cluster_threshold, is_high=True)
            results.append({
                "level": level, "touches": touches,
                "strength": min(1.0, touches / 5.0),
                "type": "swing_resistance",
                "was_resistance": True, "was_support": False,
                "is_role_reversal": False,
            })
        for level, count in sup:
            touches = self._count_touches(candles, level, cluster_threshold, is_high=False)
            results.append({
                "level": level, "touches": touches,
                "strength": min(1.0, touches / 5.0),
                "type": "swing_support",
                "was_support": True, "was_resistance": False,
                "is_role_reversal": False,
            })
        return results

    def _psychological_levels(self, candles, pair: str) -> List[Dict]:
        if not candles:
            return []
        price = candles[-1].close
        levels = []
        if price > 100:
            interval = 50.0 if price > 500 else 1.0
        elif price > 1:
            interval = 0.01
        else:
            interval = 0.001

        margin = price * 0.05
        low_bound = price - margin
        high_bound = price + margin
        n = int(low_bound / interval)
        while n * interval <= high_bound:
            lvl = round(n * interval, 6)
            if lvl > 0:
                touches = self._count_touches(candles, lvl, 0.003, is_high=False)
                touches += self._count_touches(candles, lvl, 0.003, is_high=True)
                strength = 0.3 + min(0.5, touches / 8.0)
                levels.append({
                    "level": lvl, "touches": touches, "strength": strength,
                    "type": "psychological",
                    "was_support": True, "was_resistance": True,
                    "is_role_reversal": False,
                })
            n += 1
        return levels

    def _volume_profile_levels(self, candles, cluster_threshold) -> List[Dict]:
        prices = []
        volumes = []
        for c in candles:
            if getattr(c, "volume", None):
                typical = (c.high + c.low + c.close) / 3
                prices.append(typical)
                volumes.append(c.volume)
        if not prices:
            return []

        price_min = min(prices)
        price_max = max(prices)
        bins = 50
        bin_size = (price_max - price_min) / bins if price_max > price_min else 0.0001

        vol_nodes = [0.0] * bins
        price_nodes = [price_min + (i + 0.5) * bin_size for i in range(bins)]

        for p, v in zip(prices, volumes):
            idx = min(bins - 1, int((p - price_min) / bin_size))
            vol_nodes[idx] += v

        max_vol = max(vol_nodes) if vol_nodes else 0
        if max_vol == 0:
            return []
        threshold = max_vol * 0.7

        results = []
        current_price = candles[-1].close
        for i, (vol, price_node) in enumerate(zip(vol_nodes, price_nodes)):
            if vol >= threshold:
                is_support = price_node < current_price
                touches = self._count_touches(candles, price_node, cluster_threshold, is_high=not is_support)
                strength = min(1.0, 0.4 + (vol / max_vol) * 0.6)
                results.append({
                    "level": round(price_node, 6), "touches": touches,
                    "strength": strength, "type": "volume_node",
                    "was_support": is_support, "was_resistance": not is_support,
                    "is_role_reversal": False,
                })
        return results

    def _order_block_levels(self, candles) -> List[Dict]:
        if len(candles) < 10:
            return []
        results = []
        lookback = min(len(candles) - 1, 100)
        for i in range(2, lookback):
            c = candles[-(i + 1)]
            next_c = candles[-i]
            if c.range == 0:
                continue
            body_pct = abs(c.close - c.open) / c.range
            if c.close < c.open and body_pct > 0.7:
                move = (c.open - next_c.close) / c.range
                if move > 1.0:
                    level = (c.open + c.close) / 2
                    results.append({
                        "level": round(level, 6), "touches": 1,
                        "strength": 0.65, "type": "bearish_order_block",
                        "was_resistance": True, "was_support": False,
                        "is_role_reversal": False,
                    })
            if c.close > c.open and body_pct > 0.7:
                move = (next_c.close - c.open) / c.range
                if move > 1.0:
                    level = (c.open + c.close) / 2
                    results.append({
                        "level": round(level, 6), "touches": 1,
                        "strength": 0.65, "type": "bullish_order_block",
                        "was_support": True, "was_resistance": False,
                        "is_role_reversal": False,
                    })
        return results[:8]

    def _cluster_levels(self, levels, threshold):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current = [levels[0]]
        for level in levels[1:]:
            if (level - current[0]) / current[0] < threshold:
                current.append(level)
            else:
                clusters.append((np.mean(current), len(current)))
                current = [level]
        if current:
            clusters.append((np.mean(current), len(current)))
        return clusters

    def _count_touches(self, candles, level, threshold, is_high):
        touches = 0
        for c in candles:
            if is_high:
                if abs(c.high - level) / level < threshold or abs(c.close - level) / level < threshold:
                    touches += 1
            else:
                if abs(c.low - level) / level < threshold or abs(c.close - level) / level < threshold:
                    touches += 1
        return touches

    def _merge_levels(self, levels, threshold):
        if not levels:
            return []
        levels.sort(key=lambda x: x["level"])
        merged = [levels[0]]
        for lvl in levels[1:]:
            prev = merged[-1]
            if abs(lvl["level"] - prev["level"]) / prev["level"] < threshold:
                if lvl.get("strength", 0) > prev.get("strength", 0):
                    merged[-1] = lvl
            else:
                merged.append(lvl)
        return merged

enhanced_sr = EnhancedSRDetector()
