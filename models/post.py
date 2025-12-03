# models/post.py

import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from base import Base

class Post(Base):
    __tablename__ = 'posts'

    postID = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.UserID'), nullable=False)
    title = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)
    tag = Column(String(50), nullable=False)  
    likes_count = Column(Integer, default=0)
    postDate = Column(DateTime(timezone=True), server_default=func.now())

    creator = relationship("User", back_populates="posts")
    
    def __repr__(self):
        return f"<Post(title='{self.title}', tag='{self.tag}', user_id='{self.user_id}')>"