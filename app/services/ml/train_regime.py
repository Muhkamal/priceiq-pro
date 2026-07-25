"""
PriceIQ Pro — Regime Model Trainer
Trains ML regime classifier using real market candles
"""

import asyncio
from app.services.ml.regime_classifier import RegimeClassifier
from app.services.data_fetcher import DataFetcher


async def main():
    try:
        # ✅ Initialize fetcher
        fetcher = DataFetcher()

        # ✅ Fetch candles (IMPORTANT: await)
        candles = await fetcher.get_candles(
            pair="EURUSD",
            timeframe="1h",
            limit=1000
        )

        print(f"Candles fetched: {len(candles)}")

        if not candles or len(candles) < 50:
            raise ValueError("Not enough candles fetched for training")

        # ✅ Train model (NO DataFrame — pass candles directly)
        model = RegimeClassifier()

        result = model.train(candles)

        print("\n✅ Training complete")
        print(f"Samples used: {result.get('n_samples')}")
        print(f"CV accuracy: {result.get('cv_accuracy'):.2f}")

        # ✅ Save model
        model.model.save_model("regime_model.json")
        print("💾 Model saved as regime_model.json")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
