from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PacketInfo(BaseModel):
    summary: str

@app.post("/packet")
def receive_packet(packet: PacketInfo):
    print("PACKET:", packet.summary, flush=True)
    return {"status": "received"}