import asyncio
from app.services.data_fetcher import DataFetcher
from app.services.ml.regime_classifier import RegimeClassifier


async def main():
    fetcher = DataFetcher()

    # ✅ get fresh candles
    candles = await fetcher.get_candles("EURUSD", "1h", 200)

    # ✅ load model
    model = RegimeClassifier()
    model.load("regime_model.pkl")

    # ✅ predict
    prediction = model.predict(candles)

    print("\n🧠 Model Prediction:")
    print(prediction)


if __name__ == "__main__":
    asyncio.run(main())
