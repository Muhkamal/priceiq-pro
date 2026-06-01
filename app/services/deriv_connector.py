"""
PriceIQ Pro — Deriv Broker Connector v1.0

Deriv (formerly Binary.com) uses a WebSocket API (not REST).
All communication is via JSON messages over a persistent WS connection.

Features:
  - Live price streaming (ticks)
  - Place buy/sell orders (multipliers + forex CFDs)
  - Account balance retrieval
  - Open contract tracking
  - Automatic reconnection on disconnect

Get API token: https://app.deriv.com/account/api-token
Deriv API docs: https://api.deriv.com

Supported account types:
  - Real    : wss://ws.binaryws.com/websockets/v3?app_id=YOUR_APP_ID
  - Demo    : same endpoint, demo token

Supported instruments (forex CFDs on Deriv):
  EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD,
  EURJPY, GBPJPY, XAUUSD (Gold)

Usage:
    deriv = DerivConnector(api_token="your_token", app_id="your_app_id", demo=True)
    await deriv.connect()
    balance = await deriv.get_balance()
    result  = await deriv.execute_signal(signal, stake=10.0)
    await deriv.disconnect()
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Callable
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)

# Optional websockets import
try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    logger.warning("websockets not installed — DerivConnector unavailable. Run: pip install websockets")

# Deriv WebSocket endpoint
_DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3"

# Deriv symbol mapping (forex CFDs)
_DERIV_SYMBOL_MAP = {
    "EURUSD": "frxEURUSD", "GBPUSD": "frxGBPUSD",
    "USDJPY": "frxUSDJPY", "USDCHF": "frxUSDCHF",
    "AUDUSD": "frxAUDUSD", "NZDUSD": "frxNZDUSD",
    "USDCAD": "frxUSDCAD", "EURGBP": "frxEURGBP",
    "EURJPY": "frxEURJPY", "GBPJPY": "frxGBPJPY",
    "XAUUSD": "frxXAUUSD",
}

# Contract types for forex CFDs on Deriv
_DERIV_CONTRACT_TYPES = {
    "buy":  "MULTUP",    # Multiplier UP (long)
    "sell": "MULTDOWN",  # Multiplier DOWN (short)
}


class DerivOrderResult(BaseModel):
    success:       bool
    contract_id:   Optional[str]   = None
    buy_price:     Optional[float] = None
    payout:        Optional[float] = None
    start_time:    Optional[str]   = None
    symbol:        Optional[str]   = None
    direction:     Optional[str]   = None
    stake:         Optional[float] = None
    multiplier:    Optional[int]   = None
    error:         Optional[str]   = None
    raw_response:  Optional[Dict]  = None
    executed_at:   Optional[datetime] = None

    def __init__(self, **data):
        if not data.get("executed_at"):
            data["executed_at"] = datetime.now(timezone.utc)
        super().__init__(**data)


class DerivAccountInfo(BaseModel):
    broker:        str    = "Deriv"
    account_id:    str    = ""
    balance:       float  = 0.0
    currency:      str    = "USD"
    is_demo:       bool   = True
    email:         str    = ""
    loginid:       str    = ""


class DerivConnector:
    """
    Deriv WebSocket API connector.

    Connection lifecycle:
      1. Call await connect() to establish WS connection
      2. Call await authorize() to authenticate with API token
      3. Use execute_signal(), get_balance(), stream_ticks(), etc.
      4. Call await disconnect() when done

    For production, keep connection alive and use ping/pong to maintain it.
    """

    def __init__(
        self,
        api_token: str,
        app_id:    str   = "1089",   # Deriv's default public app_id for testing
        demo:      bool  = True,
    ):
        self.api_token  = api_token
        self.app_id     = app_id
        self.demo       = demo
        self._ws        = None
        self._connected = False
        self._authorized = False
        self._pending:   Dict[str, asyncio.Future] = {}
        self._tick_callbacks: List[Callable] = []
        self._listener_task: Optional[asyncio.Task] = None

    # ──────────────────────────────────────────────────────────
    # Connection management
    # ──────────────────────────────────────────────────────────

    async def connect(self):
        """Establish WebSocket connection and authorize."""
        if not _WS_AVAILABLE:
            raise RuntimeError(
                "websockets library not installed.\n"
                "Run: pip install websockets"
            )

        url = f"{_DERIV_WS_URL}?app_id={self.app_id}"
        logger.info(f"Connecting to Deriv {'demo' if self.demo else 'live'} account...")

        self._ws = await websockets.connect(
            url,
            ping_interval=30,
            ping_timeout=10,
        )
        self._connected = True

        # Start background listener
        self._listener_task = asyncio.create_task(self._listen())

        # Authorize
        await self._authorize()
        logger.info("✅ Deriv connected and authorized")

    async def disconnect(self):
        """Close WebSocket connection cleanly."""
        if self._listener_task:
            self._listener_task.cancel()
        if self._ws:
            await self._ws.close()
        self._connected  = False
        self._authorized = False
        logger.info("Deriv disconnected")

    async def reconnect(self):
        """Reconnect after disconnect."""
        logger.info("Deriv reconnecting...")
        await asyncio.sleep(5)
        await self.connect()

    # ──────────────────────────────────────────────────────────
    # Authorization
    # ──────────────────────────────────────────────────────────

    async def _authorize(self):
        response = await self._send_and_wait({
            "authorize": self.api_token,
        })
        if "error" in response:
            raise RuntimeError(f"Deriv auth failed: {response['error']['message']}")
        self._authorized = True
        account = response.get("authorize", {})
        logger.info(f"Authorized as: {account.get('email', 'unknown')} | Balance: {account.get('balance')} {account.get('currency')}")
        return account

    # ──────────────────────────────────────────────────────────
    # Account
    # ──────────────────────────────────────────────────────────

    async def get_balance(self) -> DerivAccountInfo:
        """Fetch current account balance and info."""
        self._check_connected()
        response = await self._send_and_wait({"balance": 1, "subscribe": 0})

        if "error" in response:
            raise RuntimeError(f"Deriv balance error: {response['error']['message']}")

        balance_data = response.get("balance", {})
        return DerivAccountInfo(
            balance=float(balance_data.get("balance", 0)),
            currency=balance_data.get("currency", "USD"),
            loginid=balance_data.get("loginid", ""),
            is_demo=self.demo,
        )

    async def get_open_contracts(self) -> List[Dict]:
        """Get all currently open contracts."""
        self._check_connected()
        response = await self._send_and_wait({"portfolio": 1})
        if "error" in response:
            return []
        contracts = response.get("portfolio", {}).get("contracts", [])
        return contracts

    # ──────────────────────────────────────────────────────────
    # Trade execution
    # ──────────────────────────────────────────────────────────

    async def execute_signal(
        self,
        signal:     Any,
        stake:      float = 10.0,     # USD amount to risk
        multiplier: int   = 10,       # leverage multiplier (10x, 20x, 50x, 100x)
        stop_loss:  Optional[float] = None,   # override signal SL in USD
        take_profit: Optional[float] = None,  # override signal TP in USD
    ) -> DerivOrderResult:
        """
        Execute a trade signal on Deriv as a multiplier contract.

        Deriv multiplier contracts:
          - You stake X USD with Y multiplier
          - Your P&L = price_move% × multiplier × stake
          - Max loss = stake (built-in SL at stake level by default)

        Args:
            signal     : TradeSignal object
            stake      : Amount in USD to put on the trade
            multiplier : Leverage (10, 20, 50, 100)
            stop_loss  : Optional stop loss in USD (defaults to full stake)
            take_profit: Optional take profit in USD
        """
        self._check_connected()

        pair         = signal.pair.upper()
        deriv_symbol = _DERIV_SYMBOL_MAP.get(pair)

        if not deriv_symbol:
            return DerivOrderResult(
                success=False,
                error=f"Pair {pair} not supported on Deriv. Supported: {list(_DERIV_SYMBOL_MAP.keys())}"
            )

        direction     = signal.direction.value
        contract_type = _DERIV_CONTRACT_TYPES.get(direction)

        if not contract_type:
            return DerivOrderResult(success=False, error=f"Invalid direction: {direction}")

        # Build proposal request
        proposal_req = {
            "proposal":       1,
            "amount":         stake,
            "basis":          "stake",
            "contract_type":  contract_type,
            "currency":       "USD",
            "symbol":         deriv_symbol,
            "multiplier":     multiplier,
        }

        # Add stop loss (in USD)
        if stop_loss is not None:
            proposal_req["limit_order"] = {
                "stop_loss": stop_loss,
            }
        else:
            # Default: stop loss = full stake (max loss = what you put in)
            proposal_req["limit_order"] = {
                "stop_loss": stake,
            }

        # Add take profit (in USD)
        if take_profit is not None:
            proposal_req.setdefault("limit_order", {})["take_profit"] = take_profit

        # Step 1: Get proposal (price quote)
        logger.info(f"Getting Deriv proposal for {pair} {direction.upper()} stake=${stake} x{multiplier}...")
        proposal_resp = await self._send_and_wait(proposal_req)

        if "error" in proposal_resp:
            err = proposal_resp["error"]["message"]
            logger.error(f"Deriv proposal error for {pair}: {err}")
            return DerivOrderResult(success=False, error=f"Proposal failed: {err}", raw_response=proposal_resp)

        proposal = proposal_resp.get("proposal", {})
        proposal_id = proposal.get("id")

        if not proposal_id:
            return DerivOrderResult(success=False, error="No proposal ID returned", raw_response=proposal_resp)

        # Step 2: Buy the contract
        logger.info(f"Buying Deriv contract {proposal_id}...")
        buy_resp = await self._send_and_wait({
            "buy":   proposal_id,
            "price": proposal.get("ask_price", stake),
        })

        if "error" in buy_resp:
            err = buy_resp["error"]["message"]
            logger.error(f"Deriv buy error for {pair}: {err}")
            return DerivOrderResult(success=False, error=f"Buy failed: {err}", raw_response=buy_resp)

        buy_data    = buy_resp.get("buy", {})
        contract_id = str(buy_data.get("contract_id", ""))

        logger.info(f"✅ Deriv contract opened: {pair} {direction.upper()} contract_id={contract_id} buy_price={buy_data.get('buy_price')}")

        return DerivOrderResult(
            success=True,
            contract_id=contract_id,
            buy_price=float(buy_data.get("buy_price", 0)),
            payout=float(buy_data.get("payout", 0)),
            start_time=buy_data.get("start_time"),
            symbol=deriv_symbol,
            direction=direction,
            stake=stake,
            multiplier=multiplier,
            raw_response=buy_resp,
        )

    async def close_contract(self, contract_id: str) -> Dict:
        """
        Close (sell) an open contract before expiry.
        Returns sell response dict.
        """
        self._check_connected()
        response = await self._send_and_wait({
            "sell":  contract_id,
            "price": 0,   # 0 = sell at market price
        })
        if "error" in response:
            logger.error(f"Deriv close_contract error: {response['error']['message']}")
        return response

    async def get_contract_details(self, contract_id: str) -> Dict:
        """Get details of a specific open or closed contract."""
        self._check_connected()
        response = await self._send_and_wait({
            "proposal_open_contract": 1,
            "contract_id": int(contract_id),
        })
        return response.get("proposal_open_contract", {})

    # ──────────────────────────────────────────────────────────
    # Live price streaming
    # ──────────────────────────────────────────────────────────

    async def stream_ticks(self, pair: str, callback: Callable):
        """
        Stream live tick prices for a pair.
        Calls callback(pair, price, timestamp) on each tick.

        Args:
            pair     : e.g. "EURUSD"
            callback : async or sync function(pair, price, timestamp)
        """
        self._check_connected()
        deriv_symbol = _DERIV_SYMBOL_MAP.get(pair.upper())
        if not deriv_symbol:
            raise ValueError(f"Pair {pair} not supported on Deriv")

        self._tick_callbacks.append((pair, callback))

        await self._send({
            "ticks":     deriv_symbol,
            "subscribe": 1,
        })
        logger.info(f"✅ Subscribed to {pair} tick stream")

    async def stream_multiple_pairs(self, pairs: List[str], callback: Callable):
        """Subscribe to tick stream for multiple pairs simultaneously."""
        for pair in pairs:
            await self.stream_ticks(pair, callback)

    async def get_current_price(self, pair: str) -> Optional[float]:
        """Get current spot price for a pair (single request, no subscription)."""
        self._check_connected()
        deriv_symbol = _DERIV_SYMBOL_MAP.get(pair.upper())
        if not deriv_symbol:
            return None

        response = await self._send_and_wait({
            "ticks": deriv_symbol,
        })

        tick = response.get("tick", {})
        price = tick.get("ask") or tick.get("quote")
        return float(price) if price else None

    # ──────────────────────────────────────────────────────────
    # WebSocket internals
    # ──────────────────────────────────────────────────────────

    async def _send(self, payload: Dict):
        """Send a message without waiting for response."""
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected to Deriv. Call await connect() first.")
        req_id = str(uuid.uuid4().int)[:8]
        payload["req_id"] = int(req_id)
        await self._ws.send(json.dumps(payload))

    async def _send_and_wait(self, payload: Dict, timeout: float = 15.0) -> Dict:
        """Send a message and wait for the corresponding response."""
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected to Deriv. Call await connect() first.")

        req_id           = uuid.uuid4().int % 100000
        payload["req_id"] = req_id

        future = asyncio.get_event_loop().create_future()
        self._pending[str(req_id)] = future

        await self._ws.send(json.dumps(payload))

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(str(req_id), None)
            raise TimeoutError(f"Deriv API timeout for request {req_id}")

    async def _listen(self):
        """Background task: listen for all incoming WS messages."""
        try:
            async for raw_message in self._ws:
                try:
                    message = json.loads(raw_message)
                    req_id  = str(message.get("req_id", ""))

                    # Route to waiting future if it's a response
                    if req_id in self._pending:
                        future = self._pending.pop(req_id)
                        if not future.done():
                            future.set_result(message)

                    # Route tick data to callbacks
                    if "tick" in message:
                        tick   = message["tick"]
                        symbol = tick.get("symbol", "")
                        price  = float(tick.get("ask", tick.get("quote", 0)))
                        ts     = datetime.fromtimestamp(tick.get("epoch", 0), tz=timezone.utc)

                        for pair, callback in self._tick_callbacks:
                            if _DERIV_SYMBOL_MAP.get(pair.upper()) == symbol:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(pair, price, ts)
                                    else:
                                        callback(pair, price, ts)
                                except Exception as e:
                                    logger.error(f"Tick callback error: {e}")

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Deriv listener error: {e}")

        except Exception as e:
            logger.error(f"Deriv WS connection lost: {e}")
            self._connected = False
            if not self._ws.closed:
                await self.reconnect()

    def _check_connected(self):
        if not self._connected or not self._authorized:
            raise RuntimeError(
                "Not connected to Deriv. Call:\n"
                "    await deriv.connect()\n"
                "before making API calls."
            )

    # ──────────────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "connected":   self._connected,
            "authorized":  self._authorized,
            "demo":        self.demo,
            "app_id":      self.app_id,
            "pending_reqs": len(self._pending),
            "tick_subscriptions": len(self._tick_callbacks),
        }
