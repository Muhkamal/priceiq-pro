"""
PriceIQ Pro — Deriv Broker Adapter v1.0

Wraps DerivConnector to match the BaseBroker interface so it works
as a drop-in replacement for OANDA/Alpaca in trade_engine.py.

Usage in create_broker():
    broker = create_broker("deriv", paper=True,
                           api_token="...", app_id="...")
    await broker.connect()   # must call this before use
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from app.services.broker_connector import BaseBroker, OrderResult, AccountInfo, OrderStatus
from app.services.deriv_connector import DerivConnector, DerivOrderResult

logger = logging.getLogger(__name__)


class DerivAdapter(BaseBroker):
    """
    Adapts DerivConnector to the BaseBroker interface.

    Key differences from OANDA/Alpaca:
      - Deriv uses WebSocket (must call await connect() first)
      - Deriv uses "stake" not "units" for position sizing
      - Deriv contracts use multipliers not lot sizes
      - Stop loss / take profit are in USD (not price levels)

    Position sizing for Deriv:
      stake     = account_balance * risk_percent / 100
      max_loss  = stake (Deriv multiplier contracts cap loss at stake)
      multiplier = controls leverage (10x, 20x, 50x, 100x)

    SL/TP in USD:
      stop_loss_usd  = stake (full stake at risk by default)
      take_profit_usd = stake * (risk_reward_ratio)
    """

    def __init__(
        self,
        api_token:  str,
        app_id:     str  = "1089",
        demo:       bool = True,
        multiplier: int  = 10,     # default leverage
    ):
        self._connector  = DerivConnector(api_token=api_token, app_id=app_id, demo=demo)
        self.multiplier  = multiplier
        self.demo        = demo
        self._connected  = False

    async def connect(self):
        """Must call before any trades. Establishes WS connection."""
        await self._connector.connect()
        self._connected = True

    async def disconnect(self):
        await self._connector.disconnect()
        self._connected = False

    def _check_connected(self):
        if not self._connected:
            raise RuntimeError(
                "DerivAdapter not connected. Call await broker.connect() first."
            )

    # ──────────────────────────────────────────────────────────
    # BaseBroker interface implementation
    # ──────────────────────────────────────────────────────────

    async def execute_signal(self, signal: Any, units: int) -> OrderResult:
        """
        Execute a signal on Deriv.

        units is ignored — Deriv uses stake-based sizing.
        Stake is calculated from signal.position_size["risk_amount"].
        """
        self._check_connected()

        # Get stake from signal risk amount
        if signal.position_size and isinstance(signal.position_size, dict):
            stake = float(signal.position_size.get("risk_amount", 10.0))
        else:
            stake = 10.0  # minimum stake fallback

        stake = max(1.0, round(stake, 2))  # Deriv minimum is $1

        # Convert signal SL/TP from price levels to USD amounts
        # For multiplier contracts: P&L = price_change% × multiplier × stake
        entry = signal.entry_price
        sl    = signal.stop_loss
        tp1   = signal.take_profit_1

        if entry and sl and entry != 0:
            sl_pct         = abs(entry - sl) / entry
            stop_loss_usd  = round(sl_pct * self.multiplier * stake, 2)
        else:
            stop_loss_usd  = stake  # cap at full stake

        if entry and tp1 and entry != 0:
            tp_pct          = abs(entry - tp1) / entry
            take_profit_usd = round(tp_pct * self.multiplier * stake, 2)
        else:
            take_profit_usd = None

        try:
            result = await self._connector.execute_signal(
                signal=signal,
                stake=stake,
                multiplier=self.multiplier,
                stop_loss=stop_loss_usd,
                take_profit=take_profit_usd,
            )

            if result.success:
                return OrderResult(
                    success=True,
                    order_id=result.contract_id,
                    broker="Deriv",
                    pair=signal.pair,
                    direction=signal.direction.value,
                    entry_price=result.buy_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_1,
                    units=int(stake * 100),  # pseudo-units for compatibility
                    status=OrderStatus.FILLED,
                    raw_response=result.raw_response,
                )
            else:
                return OrderResult(
                    success=False,
                    broker="Deriv",
                    pair=signal.pair,
                    error=result.error,
                    status=OrderStatus.REJECTED,
                    raw_response=result.raw_response,
                )

        except Exception as e:
            logger.error(f"Deriv execute_signal error: {e}")
            return OrderResult(
                success=False, broker="Deriv",
                pair=signal.pair, error=str(e),
                status=OrderStatus.REJECTED,
            )

    async def close_position(self, pair: str) -> OrderResult:
        """Close all open Deriv contracts for a pair."""
        self._check_connected()
        try:
            contracts = await self._connector.get_open_contracts()
            from app.services.deriv_connector import _DERIV_SYMBOL_MAP
            deriv_symbol = _DERIV_SYMBOL_MAP.get(pair.upper())

            closed = []
            for contract in contracts:
                if contract.get("symbol") == deriv_symbol:
                    result = await self._connector.close_contract(str(contract["contract_id"]))
                    closed.append(result)

            if not closed:
                return OrderResult(success=True, broker="Deriv", pair=pair,
                                   error="No open contracts found for this pair",
                                   status=OrderStatus.CANCELLED)

            return OrderResult(
                success=True, broker="Deriv", pair=pair,
                status=OrderStatus.FILLED,
                raw_response={"closed_contracts": len(closed)},
            )

        except Exception as e:
            return OrderResult(
                success=False, broker="Deriv", pair=pair,
                error=str(e), status=OrderStatus.REJECTED,
            )

    async def get_account(self) -> AccountInfo:
        """Fetch Deriv account balance."""
        self._check_connected()
        info = await self._connector.get_balance()
        return AccountInfo(
            broker="Deriv",
            account_id=info.loginid,
            balance=info.balance,
            unrealized_pnl=0.0,     # Deriv doesn't expose this directly
            margin_used=0.0,
            margin_available=info.balance,
            open_positions=len(await self._connector.get_open_contracts()),
            currency=info.currency,
            is_paper=self.demo,
        )

    async def get_open_positions(self) -> List[Dict]:
        """Get all open Deriv contracts."""
        self._check_connected()
        return await self._connector.get_open_contracts()

    async def stream_prices(self, pairs: List[str], callback):
        """Stream live tick prices from Deriv."""
        self._check_connected()
        await self._connector.stream_multiple_pairs(pairs, callback)

    def get_status(self) -> Dict:
        return {
            **self._connector.get_status(),
            "multiplier": self.multiplier,
            "adapter":    "DerivAdapter",
        }
