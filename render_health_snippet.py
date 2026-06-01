# Add this to your main.py (it's already at your project root)
# Just add the /health route — keep all your existing code

@app.get("/health")
async def health():
    from datetime import datetime
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
