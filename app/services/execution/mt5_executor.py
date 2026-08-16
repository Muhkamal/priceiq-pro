"""
MT5 Executor — Auto-trades signals on YOUR terminal only.
Subscribers still get Telegram. This is for your personal account.
"""
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Safety: only execute if this env var is set
MT5_ENABLED = os.getenv("MT5_AUTO_TRADE", "false").lower() == "true"
MT5_MAGIC_NUMBER = int(os.getenv("MT5_MAGIC_NUMBER", "420001"))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not installed. Run: pip install MetaTrader5")


class MT5Executor:
    """
    Connects to local MT5 terminal and executes signals.
    Designed for personal use — not subscriber accounts.
    """

    SYMBOL_MAP = {
        "XAUUSD": "XAUUSD",
        "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD",
        "USDCHF": "USDCHF",
        "AUDUSD": "AUDUSD",
        "BTCUSD": "BTCUSD",
    }

    def __init__(self):
        self.connected = False
        if not MT5_ENABLED:
            logger.info("MT5 auto-trading disabled. Set MT5_AUTO_TRADE=true to enable.")
            return
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package missing.")
            return
        self._connect()

    def _connect(self):
        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("MT5: No account logged in")
            return
        logger.info(f"MT5 connected: {account_info.login} | Balance: ${account_info.balance:.2f}")
        self.connected = True

    def execute(self, result) -> Optional[Dict]:
        """
        Execute a signal immediately on MT5.
        Returns: {'ticket': int, 'price': float, 'lots': float} or None
        """
        if not MT5_ENABLED or not self.connected:
            return None

        pair = getattr(result, "pair", "")
        direction = getattr(result, "direction", "")
        entry = getattr(result, "fill_price", 0)
        sl = getattr(result, "stop_loss", 0)
        tp1 = getattr(result, "take_profit_1", 0)
        lots = getattr(result, "adjusted_lots", 0.01)

        symbol = self.SYMBOL_MAP.get(pair, pair)
        if not symbol:
            return None

        # Safety checks
        if lots < 0.01:
            lots = 0.01
        if lots > 0.10:
            logger.warning(f"MT5: Lot size {lots} capped to 0.10")
            lots = 0.10

        # Get symbol info
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error(f"MT5: Symbol {symbol} not found")
            return None

        point = sym_info.point
        digits = sym_info.digits

        # Round prices to symbol digits
        entry_r = round(entry, digits)
        sl_r = round(sl, digits)
        tp_r = round(tp1, digits)

        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": entry_r,
            "sl": sl_r,
            "tp": tp_r,
            "deviation": 10,  # 10 points slippage max
            "magic": MT5_MAGIC_NUMBER,
            "comment": f"KaymanBot_{getattr(result, 'agent_used', 'bot')}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        try:
            result_mt5 = mt5.order_send(request)
            if result_mt5.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"MT5 EXECUTED: {symbol} {direction} {lots} lots @ {entry_r} | Ticket: {result_mt5.order}")
                return {
                    "ticket": result_mt5.order,
                    "price": result_mt5.price,
                    "lots": lots,
                    "symbol": symbol,
                }
            else:
                logger.error(f"MT5 failed: {result_mt5.retcode} | {result_mt5.comment}")
                return None
        except Exception as e:
            logger.error(f"MT5 execution error: {e}")
            return None

    def close_position(self, ticket: int):
        """Close a specific position by ticket."""
        if not self.connected:
            return False
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": mt5.symbol_info_tick(pos.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask,
            "deviation": 10,
            "magic": MT5_MAGIC_NUMBER,
            "comment": "KaymanBot_close",
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def get_open_positions(self) -> list:
        """Get all KaymanBot positions."""
        if not self.connected:
            return []
        all_pos = mt5.positions_get()
        if all_pos is None:
            return []
        return [p for p in all_pos if p.magic == MT5_MAGIC_NUMBER]


mt5_executor = MT5Executor()
