"""
PriceIQ Pro — Regime Model Trainer (FIXED)
Trains ML regime classifier using real market candles
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.ml.regime_classifier import RegimeClassifier
from app.services.data_fetcher import DataFetcher


async def main():
    try:
        fetcher = DataFetcher()

        # Fetch candles for training (multi-pair is better)
        all_candles = []
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]:
            try:
                candles = await fetcher.get_candles(pair, "1h", limit=2000)
                if candles and len(candles) > 100:
                    all_candles.extend(candles)
                    print(f"✅ {pair}: {len(candles)} candles")
            except Exception as e:
                print(f"⚠️ {pair} fetch failed: {e}")

        print(f"\nTotal candles: {len(all_candles)}")

        if len(all_candles) < 200:
            raise ValueError("Not enough candles fetched for training")

        # Train model
        model = RegimeClassifier()
        result = model.train(all_candles)

        if result.get("success"):
            print(f"\n✅ Training complete")
            print(f"Samples used: {result.get('n_samples')}")
            print(f"CV accuracy: {result.get('cv_accuracy'):.2%}")
            print(f"Model saved to: {model.model_path}")
        else:
            print(f"\n❌ Training failed: {result.get('reason')}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
