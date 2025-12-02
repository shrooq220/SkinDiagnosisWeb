# في ملف database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. تعريف سلسلة الاتصال (يجب أن تكون خارج الدالة)
SQLALCHEMY_DATABASE_URL = 'postgresql://neondb_owner:npg_pg2dkfGzj0Ch@ep-aged-grass-ag79m5xf-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'

# 2. إنشاء المحرك (Engine) مع معلمات SSL
# تم دمج معلمات الاتصال التي كنت تحاول إضافتها لتحسين الاستقرار
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    # يتم استخدام pool_pre_ping للحفاظ على الاتصال نشطًا وتجنب مهلة الاتصال
    pool_pre_ping=True, 
    # تمرير معلمات الحفاظ على نشاط الاتصال (Keep-Alives) لـ Psycopg2
    connect_args={
        'sslmode': 'require',
        'keepalives': 1,
        'keepalives_idle': 30,
        'keepalives_interval': 10,
        'keepalives_count': 5,
    }
)

# 3. تعريف مولد الجلسة (SessionLocal)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. تعريف دالة get_db لإنشاء وتوفير الجلسة (ضرورية لـ auth.py)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()