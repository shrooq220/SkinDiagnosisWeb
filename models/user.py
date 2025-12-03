from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship # يجب استيراد relationship
from sqlalchemy import Column, Integer, String , Boolean 
import uuid


from  base import Base 

class User(Base):
    __tablename__ = 'users'
    user_id = Column("UserID", String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    password = Column(String)
    first_name = Column(String)
    last_name = Column(String)
    age = Column(Integer)
    phone_number = Column(String)
    profile_picture_url = Column(String)
    gender = Column(String)
  
    
   

    diagnosis_records = relationship("models.diagnosis_record.DiagnosisRecord", backref="user") 
    
    # تأكد من تعديل الـ Post أيضاً لنفس السبب
    # يُفترض أن Post موجود في models.post
    posts = relationship("models.post.Post", back_populates="creator")




   