"""
PriceIQ Pro — Train Regime Classifier on All Pairs (Internal Model Save Fix)
"""
import asyncio
import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 🛠️ BULLETPROOF PATCH: Intercept XGBoost's fit() method directly
import xgboost as xgb
_original_xgb_fit = xgb.XGBClassifier.fit

def _patched_xgb_fit(self, X, y, *args, **kwargs):
    y = np.array(y)
    if y.min() > 0:
        y = y - y.min()  # Shift [1, 2] -> [0, 1] instantly
    return _original_xgb_fit(self, X, y, *args, **kwargs)

xgb.XGBClassifier.fit = _patched_xgb_fit
print("🛠️ Patched XGBClassifier.fit: shifting labels to 0-indexed before training")

from app.services.data_fetcher import DataFetcher
from app.services.v5_orchestrator_final import init_v5

async def main():
    print("🚀 Starting PriceIQ Pro Regime Training...")
    fetcher = DataFetcher()
    v5 = init_v5()
    
    all_pairs = [
        "EURUSD", "GBPUSD", "XAUUSD", "USDCHF", "AUDUSD", "BTCUSD",  # Original 6
        "USDJPY", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY"             # 5 New Crosses
    ]
    
    all_candles = []
    for pair in all_pairs:
        print(f"📡 Fetching data for {pair}...")
        try:
            candles = await fetcher.get_candles(pair, "1h", limit=300) 
            if candles:
                all_candles.extend(candles)
                print(f"✅ Loaded {len(candles)} candles for {pair}")
        except Exception as e:
            print(f"❌ Failed to fetch {pair}: {type(e).__name__}")
            continue
            
    if not all_candles:
        print("No data fetched. Exiting.")
        return

    print(f"\n🧠 Training XGBoost Regime Classifier on {len(all_candles)} total candles...")
    try:
        result = await asyncio.to_thread(v5.regime_clf.train, all_candles)
        
        # ═══ BULLETPROOF MODEL SAVE (FIXED) ═══
        # Extract the actual XGBoost brain from inside the wrapper class
        # This prevents the 'RegimeClassifier object has no attribute get' error on Render
        internal_model = getattr(v5.regime_clf, 'model', None) or getattr(v5.regime_clf, 'clf', None)
        if internal_model is None:
            internal_model = v5.regime_clf  # Fallback
            
        with open("regime_model.pkl", "wb") as f:
            pickle.dump(internal_model, f)
        print("✅ Saved internal XGBoost model to regime_model.pkl")
       
        feats = []
        for i in range(55, len(all_candles)):
            f = v5.regime_clf.extractor.extract(all_candles[max(0, i-100):i+1])
            if f:
                feats.append(f)
        if feats:
            v5.drift_monitor.fit_baseline(feats)
            print(f"✅ Drift monitor baseline fitted on {len(feats)} samples")

        print("✅ Training complete! Result:", result)
        print("\n📌 NEXT STEP: Run the git commands below to push the new brain to Render!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
