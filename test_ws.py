import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WS!")
            # Wait for first message
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(msg)
                print(f"Received msg with time {data.get('time')}")
            except asyncio.TimeoutError:
                print("Timeout waiting for message!")
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test_ws())
