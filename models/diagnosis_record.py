import datetime
from sqlalchemy import Column, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from base import Base
import uuid # ⬅️ تأكد من استيراد uuid

class DiagnosisRecord(Base):
    __tablename__ = 'diagnosis_records'

    # ✅ الأهم: يجب أن يكون String لدعم الـ UUID
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    user_id = Column( String, ForeignKey('users.UserID')) 
    diagnosis_name = Column(String(50), index=True) 
    confidence = Column(Float)
    image_path = Column(String) 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # ✅ دالة التهيئة (Constructor) تدعم الوسائط الاختيارية
    def __init__(self, user_id, diagnosis_name, confidence, image_path, id=None, created_at=None):
        if id: 
            self.id = id
        self.user_id = user_id
        self.diagnosis_name = diagnosis_name
        self.confidence = confidence
        self.image_path = image_path
        if created_at:
            self.created_at = created_at
            
def __repr__(self):
        return f"<DiagnosisRecord(id='{self.id}', diagnosis='{self.diagnosis_name}', user_id='{self.user_id}')>"