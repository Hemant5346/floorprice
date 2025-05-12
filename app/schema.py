from pydantic import BaseModel

class FloorPriceRequest(BaseModel):
    Country: str
    Domain: str
    Browser: str
    Os: str
