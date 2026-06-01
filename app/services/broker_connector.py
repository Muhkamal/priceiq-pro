"""
PriceIQ Pro — Broker Connector v3.2
Updated with connection pooling, health checks, and graceful shutdown.

SUPPORTED BROKERS:
  - OANDA (REST API, forex CFDs)
  - Alpaca (REST API, stocks + crypto)
  - Deriv (WebSocket, forex multipliers) — requires separate adapter
"""

import httpx
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Tuple
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)

RECONCILE_WINDOW_SECONDS = 120
RECONCILE_RETRIES        = 3
RECONCILE_RETRY_DELAY    = 30


class OrderStatus(str, Enum):
    FILLED     = "filled"
    PENDING    = "pending"
    REJECTED   = "rejected"
    CANCELLED  = "cancelled"
    TIMEOUT    = "timeout"
    RECONCILED = "reconciled"


class OrderResult(BaseModel):
    success:      bool
    order_id:     Optional[str]     = None
    broker:       str               = "unknown"
    pair:         str               = ""
    direction:    str               = ""
    entry_price:  Optional[float]   = None
    stop_loss:    Optional[float]   = None
    take_profit:  Optional[float]   = None
    units:        Optional[int]     = None
    status:       OrderStatus       = OrderStatus.PENDING
    error:        Optional[str]     = None
    reconciled:   bool              = False
    raw_response: Optional[Dict]    = None
    executed_at:  Optional[datetime] = None
    commission:   Optional[float]   = None  # Track commission costs

    def __init__(self, **data):
        if not data.get("executed_at"):
            data["executed_at"] = datetime.now(timezone.utc)
        super().__init__(**data)


class AccountInfo(BaseModel):
    broker:           str
    account_id:       str
    balance:          float
    unrealized_pnl:   float
    margin_used:      float
    margin_available: float
    open_positions:   int
    currency:         str  = "USD"
    is_paper:         bool = True


class BaseBroker(ABC):
    """Abstract base class for all broker connectors."""

    @abstractmethod
    async def execute_signal(self, signal: Any, units: int) -> OrderResult:
        """Execute a trading signal."""
        pass

    @abstractmethod
    async def close_position(self, pair: str) -> OrderResult:
        """Close all positions for a pair."""
        pass

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        pass

    @abstractmethod
    async def get_open_positions(self) -> List[Dict]:
        """Get list of open positions."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Health check — verify broker connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Graceful shutdown — close connections."""
        pass

    def _format_pair_oanda(self, pair: str) -> str:
        """Convert user-format pair to OANDA format: EURUSD → EUR_USD"""
        pair = pair.upper().replace("/", "")
        return f"{pair[:3]}_{pair[3:]}" if len(pair) == 6 else pair

    def _user_pair_to_oanda(self, user_pair: str) -> str:
        """Alias for clarity — same as _format_pair_oanda."""
        return self._format_pair_oanda(user_pair)

    def _validate_units(self, units: int, max_units: int = 1_000_000) -> int:
        """Validate and clamp unit size."""
        if abs(units) > max_units:
            logger.warning(f"Unit size {units} exceeds max {max_units}, clamping")
            return max_units if units > 0 else -max_units
        return units


# ══════════════════════════════════════════════════════════════
# OANDA BROKER
# ══════════════════════════════════════════════════════════════

class OANDABroker(BaseBroker):
    """
    OANDA fxTrade REST API v20.

    Pair formats:
      User format : "EURUSD"  (signals, logs, DB)
      OANDA format: "EUR_USD" (all API calls)
    """

    def __init__(self, api_key: str, account_id: str, paper: bool = True):
        self.api_key    = api_key
        self.account_id = account_id
        self.paper      = paper
        self.base_url   = (
            "https://api-fxpractice.oanda.com" if paper
            else "https://api-fxtrade.oanda.com"
        )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create reusable HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self):
        """Graceful shutdown — close connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        logger.info("OANDA broker connection closed")

    async def is_connected(self) -> bool:
        """Lightweight health check — ping account endpoint."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
            client = await self._get_client()
            response = await client.get(url, headers=self.headers, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def execute_signal(self, signal: Any, units: int) -> OrderResult:
        """Place market order with SL/TP and reconciliation on timeout."""
        oanda_pair = self._format_pair_oanda(signal.pair)
        units = self._validate_units(units)

        if signal.direction.value == "sell":
            units = -abs(units)
        else:
            units = abs(units)

        order_body = {
            "order": {
                "type":        "MARKET",
                "instrument":  oanda_pair,
                "units":       str(units),
                "timeInForce": "FOK",
                "stopLossOnFill": {
                    "price":       str(round(signal.stop_loss, 5)),
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price":       str(round(signal.take_profit_1, 5)),
                    "timeInForce": "GTC",
                },
            }
        }

        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        submitted_at = datetime.now(timezone.utc)

        try:
            client = await self._get_client()
            response = await client.post(url, headers=self.headers, json=order_body)
            data = response.json()

            if response.status_code in (200, 201):
                fill = data.get("orderFillTransaction", {})
                commission = float(fill.get("commission", 0)) if "commission" in fill else None
                return OrderResult(
                    success=True,
                    order_id=fill.get("id"),
                    broker="OANDA",
                    pair=signal.pair,
                    direction=signal.direction.value,
                    entry_price=float(fill.get("price", signal.entry_price)),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_1,
                    units=abs(units),
                    status=OrderStatus.FILLED,
                    raw_response=data,
                    commission=commission,
                )
            else:
                err = data.get("errorMessage", str(data))
                logger.error(f"OANDA rejected order for {signal.pair}: {err}")
                return OrderResult(
                    success=False, broker="OANDA", pair=signal.pair,
                    error=f"OANDA rejected: {err}",
                    status=OrderStatus.REJECTED, raw_response=data,
                )

        except httpx.TimeoutException:
            return await self._handle_timeout(signal, oanda_pair, units, submitted_at)
        except Exception as e:
            logger.error(f"OANDA execute_signal error for {signal.pair}: {e}")
            return OrderResult(
                success=False, broker="OANDA", pair=signal.pair,
                error=str(e), status=OrderStatus.REJECTED,
            )

    async def _handle_timeout(self, signal, oanda_pair, units, submitted_at) -> OrderResult:
        """Handle timeout with reconciliation retries."""
        logger.warning(
            f"OANDA order timed out for {signal.pair} — "
            f"starting reconciliation (up to {RECONCILE_RETRIES} attempts)..."
        )

        for attempt in range(1, RECONCILE_RETRIES + 1):
            if attempt > 1:
                logger.info(f"Reconciliation attempt {attempt}/{RECONCILE_RETRIES} for {signal.pair}...")
                await asyncio.sleep(RECONCILE_RETRY_DELAY)

            reconciled = await self._reconcile_order(
                oanda_pair=oanda_pair,
                direction=signal.direction.value,
                units=abs(units),
                submitted_at=submitted_at,
                window_seconds=RECONCILE_WINDOW_SECONDS,
            )

            if reconciled:
                order_id, fill_price = reconciled
                logger.info(f"✅ Reconciled (attempt {attempt}): {signal.pair} order {order_id} filled at {fill_price}")
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    broker="OANDA",
                    pair=signal.pair,
                    direction=signal.direction.value,
                    entry_price=fill_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_1,
                    units=abs(units),
                    status=OrderStatus.RECONCILED,
                    reconciled=True,
                )

        logger.error(f"All {RECONCILE_RETRIES} reconciliation attempts failed for {signal.pair}")
        return OrderResult(
            success=False,
            broker="OANDA",
            pair=signal.pair,
            error=f"Order timed out. {RECONCILE_RETRIES} reconciliation attempts found no fill.",
            status=OrderStatus.TIMEOUT,
        )

    async def _reconcile_order(
        self, 
        oanda_pair: str, 
        direction: str, 
        units: int, 
        submitted_at: datetime, 
        window_seconds: int
    ) -> Optional[Tuple[str, float]]:
        """Poll transactions and open trades for order fill."""
        cutoff = submitted_at - timedelta(seconds=10)

        # Check 1: Recent transactions
        tx_url = f"{self.base_url}/v3/accounts/{self.account_id}/transactions"
        try:
            client = await self._get_client()
            response = await client.get(tx_url, headers=self.headers, params={"count": 30}, timeout=15.0)
            if response.status_code == 200:
                for tx in response.json().get("transactions", []):
                    result = self._match_transaction(tx, oanda_pair, direction, cutoff)
                    if result:
                        return result
        except Exception as e:
            logger.warning(f"Reconciliation tx poll error: {e}")

        # Check 2: Open trades
        trades_url = f"{self.base_url}/v3/accounts/{self.account_id}/openTrades"
        try:
            client = await self._get_client()
            response = await client.get(trades_url, headers=self.headers, timeout=15.0)
            if response.status_code == 200:
                for trade in response.json().get("trades", []):
                    result = self._match_open_trade(trade, oanda_pair, direction, cutoff)
                    if result:
                        return result
        except Exception as e:
            logger.warning(f"Reconciliation open trades poll error: {e}")

        return None

    def _match_transaction(self, tx: Dict, oanda_pair: str, direction: str, cutoff: datetime) -> Optional[Tuple[str, float]]:
        """Match transaction against expected order."""
        if tx.get("type", "") not in ("ORDER_FILL", "MARKET_ORDER_TRANSACTION"):
            return None
        if tx.get("instrument", "") != oanda_pair:
            return None
        try:
            tx_time = datetime.fromisoformat(tx.get("time", "").replace("Z", "+00:00"))
            if tx_time < cutoff:
                return None
        except (ValueError, AttributeError):
            return None
        tx_units = int(tx.get("units", 0))
        price = float(tx.get("price", 0))
        if direction == "buy" and tx_units > 0 and price > 0:
            return str(tx.get("id", "reconciled")), price
        if direction == "sell" and tx_units < 0 and price > 0:
            return str(tx.get("id", "reconciled")), price
        return None

    def _match_open_trade(self, trade: Dict, oanda_pair: str, direction: str, cutoff: datetime) -> Optional[Tuple[str, float]]:
        """Match open trade against expected order."""
        if trade.get("instrument", "") != oanda_pair:
            return None
        try:
            open_time = datetime.fromisoformat(trade.get("openTime", "").replace("Z", "+00:00"))
            if open_time < cutoff:
                return None
        except (ValueError, AttributeError):
            return None
        trade_units = int(trade.get("initialUnits", 0))
        price = float(trade.get("price", 0))
        if direction == "buy" and trade_units > 0 and price > 0:
            return str(trade.get("id", "reconciled")), price
        if direction == "sell" and trade_units < 0 and price > 0:
            return str(trade.get("id", "reconciled")), price
        return None

    async def close_position(self, pair: str) -> OrderResult:
        """Close position for a pair."""
        oanda_pair = self._format_pair_oanda(pair)
        url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{oanda_pair}/close"
        try:
            client = await self._get_client()
            response = await client.put(
                url, headers=self.headers,
                json={"longUnits": "ALL", "shortUnits": "ALL"}
            )
            data = response.json()
            success = response.status_code == 200
            return OrderResult(
                success=success, broker="OANDA", pair=pair,
                status=OrderStatus.FILLED if success else OrderStatus.REJECTED,
                error=data.get("errorMessage") if not success else None,
                raw_response=data,
            )
        except httpx.TimeoutException:
            logger.error(f"close_position timed out for {pair} — VERIFY IN OANDA DASHBOARD")
            return OrderResult(
                success=False, broker="OANDA", pair=pair,
                error="Close position timed out — verify in OANDA dashboard",
                status=OrderStatus.TIMEOUT,
            )
        except Exception as e:
            return OrderResult(
                success=False, broker="OANDA", pair=pair,
                error=str(e), status=OrderStatus.REJECTED,
            )

    async def get_account(self) -> AccountInfo:
        url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
        client = await self._get_client()
        response = await client.get(url, headers=self.headers, timeout=10.0)
        data = response.json()
        acct = data.get("account", {})
        return AccountInfo(
            broker="OANDA", account_id=self.account_id,
            balance=float(acct.get("balance", 0)),
            unrealized_pnl=float(acct.get("unrealizedPL", 0)),
            margin_used=float(acct.get("marginUsed", 0)),
            margin_available=float(acct.get("marginAvailable", 0)),
            open_positions=int(acct.get("openPositionCount", 0)),
            currency=acct.get("currency", "USD"),
            is_paper=self.paper,
        )

    async def get_open_positions(self) -> List[Dict]:
        url = f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"
        client = await self._get_client()
        response = await client.get(url, headers=self.headers, timeout=10.0)
        return response.json().get("positions", [])

    async def stream_prices(self, pairs: List[str], callback):
        """Stream prices with automatic reconnection."""
        instruments = ",".join(self._format_pair_oanda(p) for p in pairs)
        stream_base = (
            "https://stream-fxpractice.oanda.com" if self.paper
            else "https://stream-fxtrade.oanda.com"
        )
        url = f"{stream_base}/v3/accounts/{self.account_id}/pricing/stream?instruments={instruments}"

        while True:
            try:
                logger.info(f"Connecting to OANDA price stream for {pairs}...")
                client = await self._get_client()
                async with client.stream("GET", url, headers=self.headers, timeout=None) as response:
                    if response.status_code != 200:
                        logger.error(f"Price stream returned {response.status_code}")
                        await asyncio.sleep(30)
                        continue

                    logger.info("Price stream connected")
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            tick = json.loads(line)
                            if tick.get("type") == "PRICE":
                                await callback(
                                    pair=tick["instrument"].replace("_", ""),
                                    bid=float(tick["bids"][0]["price"]),
                                    ask=float(tick["asks"][0]["price"]),
                                    timestamp=tick["time"],
                                )
                        except (json.JSONDecodeError, KeyError):
                            continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Price stream error: {e} — reconnecting in 30s")
                await asyncio.sleep(30)


# ══════════════════════════════════════════════════════════════
# ALPACA BROKER
# ══════════════════════════════════════════════════════════════

class AlpacaBroker(BaseBroker):
    """Alpaca REST API broker connector for stocks and crypto."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.paper      = paper
        self.base_url   = (
            "https://paper-api.alpaca.markets" if paper
            else "https://api.alpaca.markets"
        )
        self.headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type":        "application/json",
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        logger.info("Alpaca broker connection closed")

    async def is_connected(self) -> bool:
        try:
            client = await self._get_client()
            r = await client.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def execute_signal(self, signal: Any, units: int) -> OrderResult:
        symbol = signal.pair.replace("/", "")
        side = "buy" if signal.direction.value == "buy" else "sell"
        units = self._validate_units(units, max_units=100000)  # Alpaca has lower limits
        
        body = {
            "symbol": symbol, 
            "qty": str(abs(units)), 
            "side": side,
            "type": "market", 
            "time_in_force": "gtc", 
            "order_class": "bracket",
            "stop_loss": {"stop_price": str(round(signal.stop_loss, 5))},
            "take_profit": {"limit_price": str(round(signal.take_profit_1, 5))},
        }
        try:
            client = await self._get_client()
            response = await client.post(f"{self.base_url}/v2/orders", headers=self.headers, json=body)
            data = response.json()
            if response.status_code in (200, 201):
                return OrderResult(
                    success=True, order_id=data.get("id"), broker="Alpaca",
                    pair=signal.pair, direction=signal.direction.value,
                    entry_price=float(data.get("filled_avg_price") or signal.entry_price),
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit_1,
                    units=abs(units),
                    status=OrderStatus.FILLED if data.get("status") == "filled" else OrderStatus.PENDING,
                    raw_response=data,
                )
            return OrderResult(
                success=False, broker="Alpaca", pair=signal.pair,
                error=data.get("message", str(data)), status=OrderStatus.REJECTED, raw_response=data,
            )
        except httpx.TimeoutException:
            logger.warning(f"Alpaca order timed out for {signal.pair}")
            return OrderResult(
                success=False, broker="Alpaca", pair=signal.pair,
                error="Order timed out — verify in Alpaca dashboard",
                status=OrderStatus.TIMEOUT,
            )
        except Exception as e:
            return OrderResult(success=False, broker="Alpaca", pair=signal.pair, error=str(e), status=OrderStatus.REJECTED)

    async def close_position(self, pair: str) -> OrderResult:
        try:
            client = await self._get_client()
            r = await client.delete(f"{self.base_url}/v2/positions/{pair.replace('/', '')}", headers=self.headers)
            return OrderResult(
                success=r.status_code == 200, broker="Alpaca", pair=pair,
                status=OrderStatus.FILLED if r.status_code == 200 else OrderStatus.REJECTED,
                error=r.text if r.status_code != 200 else None,
            )
        except Exception as e:
            return OrderResult(success=False, broker="Alpaca", pair=pair, error=str(e), status=OrderStatus.REJECTED)

    async def get_account(self) -> AccountInfo:
        client = await self._get_client()
        r = await client.get(f"{self.base_url}/v2/account", headers=self.headers)
        d = r.json()
        return AccountInfo(
            broker="Alpaca", account_id=d.get("account_number", ""),
            balance=float(d.get("cash", 0)), unrealized_pnl=float(d.get("unrealized_pl", 0)),
            margin_used=float(d.get("initial_margin", 0)), margin_available=float(d.get("buying_power", 0)),
            open_positions=int(float(d.get("position_market_value", 0)) > 0), 
            currency="USD", is_paper=self.paper,
        )

    async def get_open_positions(self) -> List[Dict]:
        client = await self._get_client()
        r = await client.get(f"{self.base_url}/v2/positions", headers=self.headers)
        return r.json() if r.status_code == 200 else []


# ══════════════════════════════════════════════════════════════
# BROKER FACTORY
# ══════════════════════════════════════════════════════════════

def create_broker(broker_type: str, paper: bool = True, **kwargs) -> BaseBroker:
    """
    Factory function to create a broker connector.

    Supported:
      "oanda"  → OANDABroker  (REST API, forex CFDs)
      "alpaca" → AlpacaBroker (REST API, stocks + crypto)
      "deriv"  → DerivAdapter (WebSocket, forex multipliers)
                 NOTE: call await broker.connect() after creation

    Examples:
        broker = create_broker("oanda",  paper=True, api_key="...", account_id="...")
        broker = create_broker("alpaca", paper=True, api_key="...", api_secret="...")
        broker = create_broker("deriv",  paper=True, api_token="...", app_id="...")
        await broker.connect()  # required for Deriv only
    """
    broker_type = broker_type.lower()

    if broker_type == "oanda":
        return OANDABroker(
            api_key=kwargs["api_key"],
            account_id=kwargs["account_id"],
            paper=paper
        )
    elif broker_type == "alpaca":
        return AlpacaBroker(
            api_key=kwargs["api_key"],
            api_secret=kwargs["api_secret"],
            paper=paper
        )
    elif broker_type == "deriv":
        try:
            from app.services.deriv_adapter import DerivAdapter
            return DerivAdapter(
                api_token=kwargs["api_token"],
                app_id=kwargs.get("app_id", "1089"),
                demo=paper,
                multiplier=kwargs.get("multiplier", 10),
            )
        except ImportError as e:
            raise ImportError(
                f"Deriv adapter not available: {e}\n"
                "Make sure deriv_adapter.py and deriv_connector.py exist in app/services/"
            )
    else:
        raise ValueError(
            f"Unknown broker: '{broker_type}'. "
            f"Use 'oanda', 'alpaca', or 'deriv'."
        )


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

async def get_broker_health(broker: BaseBroker) -> Dict[str, Any]:
    """Get health status of a broker connection."""
    try:
        connected = await broker.is_connected()
        account = await broker.get_account() if connected else None
        return {
            "connected": connected,
            "broker": account.broker if account else "unknown",
            "paper": account.is_paper if account else None,
            "balance": account.balance if account else None,
            "open_positions": account.open_positions if account else None,
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
        }
