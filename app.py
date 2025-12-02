import psycopg2
import psycopg2.extras 
from flask import (
    Flask, jsonify, request, render_template, 
    send_from_directory, url_for, redirect, session, g
)
from flask_cors import CORS
import os
# Assumed project files and modules
from auth import auth_blueprint
from auth import auth_middleware
from database import get_db, engine 
from werkzeug.utils import secure_filename
import jwt 
import uuid 
from models.diagnosis_record import DiagnosisRecord
from models.user import User 
from models.post import Post 
from base import Base
from werkzeug.datastructures import FileStorage
from io import BytesIO 
import datetime



BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# ==================================
# Translation Data Configuration
# ==================================
TRANSLATIONS_DATA = {
    'ar': {
        "history": "السجل",
        "community": "المجتمع",
        "about": "عنا",
        "create": "إنشاء منشور جديد",
        "back": "<i class='fas fa-arrow-left'></i> العودة للمجتمع",
        "tag": "التصنيف <span class='req'>*</span>", 
        "none": "لا شيء",
        "melanoma": "ميلانوما",
        "lupus": "ذئبة",
        "ringworm": "سعفة/فطار جلدي",
        "eczema": "إكزيما",
        "acne": "حب شباب",
        "title": "العنوان <span class='req'>*</span>",
        "body": "نص المنشور <span class='req'>*</span>",
        "titleInput": "اكتب عنواناً واضحاً ومفيداً",
        "bodyInput": "شارك سؤالك أو تجربتك...",
        "help": "نصيحة المجتمع ليست تشخيصاً طبياً. تجنب المعرفات الشخصية.",
        "guidelines": "أوافق على إرشادات المجتمع <span class='req'>*</span>",
        "cancel": "إلغاء",
        "publish": "نشر المنشور"
    },
    'en': {
        "history": "History",
        "community": "Community",
        "about": "About",
        "create": "Create a New Post",
        "back": "<i class='fas fa-arrow-left'></i> Back to Community",
        "tag": "Tag <span class='req'>*</span>", 
        "none": "None",
        "melanoma": "Melanoma",
        "lupus": "Lupus",
        "ringworm": "Ringworm",
        "eczema": "Eczema",
        "acne": "Acne",
        "title": "Title <span class='req'>*</span>",
        "body": "Post Body <span class='req'>*</span>",
        "titleInput": "Write a clear, helpful title",
        "bodyInput": "Share your question or experience...",
        "help": "Community advice is not medical diagnosis. Avoid personal identifiers.",
        "guidelines": "I agree to the community guidelines <span class='req'>*</span>",
        "cancel": "Cancel",
        "publish": "Publish Post"
    }
}

# ==================================
# PostgreSQL Database Configuration
# ==================================
DATABASE_URL = 'postgresql://neondb_owner:npg_pg2dkfGzj0Ch@ep-aged-grass-ag79m5xf-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'


# ==================================
# Application and Blueprint Setup
# ==================================
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates') 
app.secret_key = '1hdfkhrtfd@#d356hsy'



CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads' 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) 
app.register_blueprint(auth_blueprint, url_prefix='/auth')


# ==================================
# 💡 وضع كود إنشاء المجلد وتهيئة Flask هنا
# ==================================
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ======================
# AI Model Initialization
# ======================
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_base

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_final.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'])

# Global confidence threshold
CONFIDENCE_THRESHOLD = 40.0 # Minimum confidence percentage required for diagnosis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MODEL_LOADED = False

if os.path.exists(MODEL_PATH):
    try:
        ck = torch.load(MODEL_PATH, map_location=device)
        num_classes = ck["num_classes"]
        class_names = ck["class_names"]
        img_size    = ck["img_size"]
        mean, std   = ck["mean"], ck["std"]

        model = convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(device).eval()
        
        # Data augmentation transformations for TTA
        base_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        hflip = transforms.RandomHorizontalFlip(p=1.0)
        vflip = transforms.RandomVerticalFlip(p=1.0)
        rot15 = transforms.RandomRotation(15)

        @torch.no_grad()
        def predict_tta_pil(pil_img):
            """
            Performs TTA (5-views) prediction using Softmax mean. 
            Returns the predicted class name and confidence (float from 0.0 to 1.0).
            """
            x = base_tf(pil_img.convert("RGB"))
            views = [x, hflip(x.clone()), vflip(x.clone()), rot15(x.clone()), hflip(rot15(x.clone()))]
            X = torch.stack(views, 0).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=='cuda')):
                 logits = model(X)
            probs = torch.softmax(logits, 1).mean(0)
            conf, idx = float(probs.max().item()), int(probs.argmax().item())
            return class_names[idx], conf 
        
        IS_MODEL_LOADED = True
    except Exception as e:
        print(f"Model loading or setup failed: {e}")
else:
    print(f"Warning: Model file not found at: {MODEL_PATH}. Prediction functionality will be disabled.")

# Fallback function if model fails to load
if not IS_MODEL_LOADED:
    # Full class names used for prediction
    class_names = ["Acne and Rosacea Photos", "Eczema Photos", "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles", "Tinea Ringworm Candidiasis and other Fungal Infections"]
    def predict_tta_pil(pil_img):
        # Returns a random result with low confidence to simulate failure
        return class_names[0], 0.10 # Acne, 10% (0.10)

def slugify_class(name: str) -> str:
    """Converts a full class name to a URL-friendly slug."""
    mapping = {
        # Maps full diagnosis names to shorter, URL-friendly slugs
        "Acne and Rosacea Photos": "Acne",
        "Eczema Photos": "Eczema",
        "Lupus and other Connective Tissue diseases": "Lupus",
        "Melanoma Skin Cancer Nevi and Moles": "Melanoma",
        "Tinea Ringworm Candidiasis and other Fungal Infections": "Ringworm", 
        "Tinea/Ringworm ": "Ringworm", 
    }
    return mapping.get(name, "Acne") 

# ==================================
# PostgreSQL Connection Management Utilities
# ==================================

def get_db_connection():
    """Establishes a database connection and stores it in Flask's g object."""
    if 'db_conn' not in g:
        g.db_conn = psycopg2.connect(DATABASE_URL)
    return g.db_conn

@app.teardown_appcontext
def close_db_connection(exception):
    """Automatically closes the database connection after each request."""
    db = g.pop('db_conn', None)
    if db is not None:
        db.close()

# -------------------------------------------------------------
# Database Table Creation Function
# -------------------------------------------------------------
def create_db_tables():
    """Creates/updates database tables based on SQLAlchemy models."""
    print("Attempting to create/update database tables...")
    try:
        # Base and engine are assumed to be correctly defined in database.py
        Base.metadata.create_all(bind=engine)
        print("Database tables created/updated successfully.")
    except Exception as e:
        print(f"Error creating database tables: {e}")

# ==================================
# Main Page Routes
# ==================================

@app.route('/')
def index():
    """Handles the root URL, redirects logged-in users to diagnosis page."""
    if 'user_id' in session:
        return redirect(url_for('diagnosis'))
    return render_template('login.html')

@app.route('/login', methods=['GET'])
def login():
    """Renders the login page, redirects logged-in users to diagnosis page."""
    if 'user_id' in session:
        return redirect(url_for('diagnosis'))
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/analysis')
def analysis():
    return render_template('Analysis.html')

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot-password.html')


@app.route('/diagnosis')
def diagnosis():
    return render_template('Diagnosis.html')

@app.route('/community')
def community():
    lang_from_request = 'en'  
    translations_data = TRANSLATIONS_DATA

    return render_template(
        'community.html',
        translations=translations_data, 
        lang=lang_from_request
    )

@app.route('/profile')
@auth_middleware
def profile(user_id_from_token=None):
    user_id = session.get('user_id') or user_id_from_token
    
    if not user_id:
        return redirect(url_for('login'))
    
    db = next(get_db())
    user_db = db.query(User).filter(User.user_id == user_id).first() 
    db.close() 
    if not user_db:
        session.pop('user_id', None) 
        return redirect(url_for('login'))
    
    # Prepare user data for the template
    full_name = f"{user_db.first_name or ''} {user_db.last_name or ''}".strip()

    return render_template('profile.html', user={
        'first_name': user_db.first_name, 
        'last_name': user_db.last_name,
        'email': user_db.email,
        'age': user_db.age,
        'phone_number': user_db.phone_number,
        'gender': user_db.gender,
        'avatar_url': url_for('uploaded_file', filename=user_db.profile_picture_url) if user_db.profile_picture_url 
        else url_for('static', filename='default_avatar.png')
    })


@app.route('/history')
def history():
    """Renders the diagnosis history page."""
    return render_template('history.html')

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('About.html')

@app.route('/unknown')
def unknown_result_page():
    """Renders the result page for unconfirmed diagnosis (low confidence)."""
    return render_template('UnknownResult.html')


@app.route('/logout')
def logout():
   session.clear()
   return redirect(url_for('login'))

# ==================================
# API ROUTES
# ==================================

@app.route('/api/history', methods=['GET'])
@auth_middleware
def api_history(user_id_from_token):
    """
    API endpoint to retrieve the diagnosis history for the authenticated user.
    Requires user authentication via the auth_middleware.
    """
    db = next(get_db())
    try:
        # Retrieve all diagnosis records for the user, ordered by creation time
        records = db.query(DiagnosisRecord).filter(DiagnosisRecord.user_id == user_id_from_token).order_by(DiagnosisRecord.created_at.desc()).all()

        results = []
        for record in records:
            # 1. Extract the filename from the full path
            image_filename = record.image_path.split('/')[-1]
            
            # Use the uploaded_file route to serve the image
            image_url = url_for('uploaded_file', filename=image_filename)
            
            # Convert confidence float (0.0-1.0) to a percentage string
            confidence_percentage = f"{record.confidence * 100:.2f}"
            
            # 2. Build the detailed URL with image and confidence parameters
            details_url = url_for(
                'disease_page', 
                disease=slugify_class(record.diagnosis_name),
                img=image_filename,  # Add image filename
                conf=confidence_percentage # Add confidence percentage
            )
            
            results.append({
                'id': record.id,
                'diagnosis_name': record.diagnosis_name,
                'confidence': f"{int(record.confidence * 100)}%",  # Display confidence as an integer percentage
                'image_url': image_url,
                'date': record.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'details_url': details_url
            })

        return jsonify({'success': True, 'history': results}), 200

    except Exception as e:
        print(f"Error retrieving history for user {user_id_from_token}: {e}")
        return jsonify({'success': False, 'error': 'An internal error occurred while fetching history.'}), 500
    finally:
        db.close()



@app.route('/create-post', methods=['GET', 'POST'])
def create_post():
    """Renders the page for creating a new community post."""
    current_lang = 'ar' 
    
    return render_template(
        'create-post.html', 
        translations=TRANSLATIONS_DATA,  
        lang=current_lang               
    )

@app.route('/api/posts', methods=['POST'])
@auth_middleware 
def create_new_post(user_id_from_token):
    """
    API endpoint for creating a new post.
    Requires authentication via auth_middleware.
    """
    db = next(get_db())
    try:
        data = request.get_json()
        
        title = data.get('title')
        body = data.get('body')
        tag = data.get('tag')
        
        if not title or not body or not tag or tag.lower() == 'none':
            return jsonify({'success': False, 'error': 'Title, body, and tag are required.'}), 400

        new_post = Post(
            user_id=user_id_from_token,
            title=title,
            body=body,
            tag=tag,
        )

        db.add(new_post)
        db.commit()

        return jsonify({'success': True, 'message': 'Post created successfully.', 'post_id': new_post.postID}), 201

    except Exception as e:
        db.rollback()
        print(f"Error creating post: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        db.close()

@app.route('/api/posts', methods=['GET'])
def get_all_posts():
    """
    API endpoint to retrieve all community posts with creator details.
    Does not require authentication.
    """
    db = next(get_db())
    try:
        # Join Posts with Users to get creator information
        posts_with_users = db.query(Post).join(User, Post.user_id == User.user_id).order_by(Post.postDate.desc()).all()
        
        results = []
        for post in posts_with_users:
            user = post.creator 
            
            creator_name = 'Deleted User'
            profile_pic_url = None
            
            if user:
                creator_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
                if not creator_name: 
                    creator_name = 'Anonymous'
                    
                if user.profile_picture_url:
                    profile_pic_url = url_for('uploaded_file', filename=user.profile_picture_url, _external=False)

            results.append({
                'id': post.postID,
                'title': post.title,
                'body': post.body,
                'tag': post.tag,
                'likes_count': post.likes_count,
                'created_at': post.postDate.isoformat(), 
                
                'user_id': post.user_id, 
                'creator_name': creator_name,
                'profile_pic_url': profile_pic_url,
            })
        
        return jsonify({'success': True, 'posts': results}), 200

    except Exception as e:
        print(f"Error fetching posts: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred while fetching posts.'}), 500
    finally:
        db.close()

# -------------------------------------------------------------
# Analysis and Redirection Logic
# -------------------------------------------------------------
@app.route('/api/analysis', methods=['POST'])
@auth_middleware 
def api_analysis(user_id_from_token):
   
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided.'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file.'}), 400
    
    db = next(get_db())
    try:
        # 1. Prediction using the AI model
        pil_img = Image.open(image_file).convert("RGB")
        # confidence_float is between 0.0 and 1.0
        diagnosis_name, confidence_float = predict_tta_pil(pil_img) 
        
        confidence_percent = confidence_float * 100 # Convert to percentage (0.0 - 100.0)

        # Confidence Threshold Check (40%)
        if confidence_percent < CONFIDENCE_THRESHOLD:
            # Redirect to the "UnknownResult" page
            details_url = url_for("unknown_result_page")
            
            # Return success=True with the redirect URL for client-side handling
            return jsonify({
                'success': True, 
                'diagnosis_name': "Unknown", 
                'confidence': round(confidence_percent, 2), 
                'details_url': details_url
            }), 200
            
        # 2. Save the file (only if confidence is sufficient)
        image_file.seek(0)
        filename = f"{uuid.uuid4()}.{secure_filename(image_file.filename).split('.')[-1]}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        image_file.save(save_path)


        # 3. Save the record in the database
        # Shorten diagnosis name for the VARCHAR(50) column
        db_diagnosis_name = diagnosis_name
        if diagnosis_name == "Tinea Ringworm Candidiasis and other Fungal Infections":
            # Use the shorter value for DB
            db_diagnosis_name = "Tinea/Ringworm " 

        # Determine the URL slug based on the diagnosis name
        slug = slugify_class(db_diagnosis_name) 
        
        new_record = DiagnosisRecord(
            id=str(uuid.uuid4()),
            user_id=user_id_from_token,
            diagnosis_name=db_diagnosis_name, # Use the safe DB name
            confidence=confidence_float, # Save the float value in the DB
            image_path=filename,
            created_at=datetime.datetime.now()
        )
        
        db.add(new_record)
        db.commit()

        # 4. Return the result to the client
        # Use the slug (e.g., Ringworm) for the result page URL
        return jsonify({
            'success': True,
            'diagnosis': diagnosis_name, # Return full name in API response
            'confidence': round(confidence_percent, 2), 
            'details_url': url_for('disease_page', disease=slug, img=filename, conf=round(confidence_percent, 2))
        }), 200

    except Exception as e:
        db.rollback()
        print(f"Error during analysis or database save: {e}")
        return jsonify({'success': False, 'message': f'Analysis failed due to a server error: {e}'}), 500
    finally:
        db.close()


# -------------------------------------------------------------
# Dynamic Route for Disease Result Pages
# -------------------------------------------------------------
@app.route("/disease/<disease>")
def disease_page(disease):
    """
    Dynamic route to render the specific disease result page.
    Renders the appropriate HTML template based on the 'disease' slug.
    """
    # Map slugs to template filenames
    templates_map = {
        "Acne":     "Acne.html",
        "Eczema":   "Eczema.html",
        "Lupus":    "Lupus.html",
        "Melanoma": "Melanoma.html",
        "Ringworm": "Ringworm.html", 
    }
    tpl = templates_map.get(disease, f"{disease}.html") 

    conf = request.args.get("conf", type=float, default=None)
    img  = request.args.get("img", type=str, default=None)

    image_url = None
    if img:
        image_url = url_for("uploaded_file", filename=img)

    # Prepare a user-friendly name for the template
    pretty_name = {
        "Acne": "Acne ",
        "Eczema": "Eczema ",
        "Lupus": "Lupus ",
        "Melanoma": "Melanoma Skin Cancer ",
        "Ringworm": "Tinea Ringworm ",
    }.get(disease, disease)
    
    return render_template(tpl,
                           pred_name=pretty_name,
                           confidence=conf,
                           image_url=image_url)




# -------------------------------------------------------------
# Route for Serving Uploaded Files
# -------------------------------------------------------------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Route to serve uploaded files from the UPLOAD_FOLDER directory."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------------------------------------------------
# Main Execution Block
# -------------------------------------------------------------
if __name__ == '__main__':
    create_db_tables()
    app.run(host="0.0.0.0", port=5000, debug=True)