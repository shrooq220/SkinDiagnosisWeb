from pydantic import BaseModel
from typing import Optional, List

class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    age: Optional[int] = None    
    phone: Optional[str] = None
    station: str
    category: str
    gender: str
    isAdmin: Optional[int] = 0
    station_access: Optional[List[str]] = None
    qualifications: Optional[List[str]] = None