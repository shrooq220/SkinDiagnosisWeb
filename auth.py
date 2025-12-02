import os
import uuid
import bcrypt
import jwt
from functools import wraps
from flask import Blueprint, jsonify, request, send_from_directory, render_template, current_app, session, url_for, redirect
from werkzeug.utils import secure_filename
from database import get_db
from models.user import User 
from werkzeug.security import generate_password_hash, check_password_hash
from models.post import Post 
auth_blueprint = Blueprint('auth', __name__)

def auth_middleware(f):
    """
    Decorator for authentication. Checks for JWT in headers or user ID in session.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Checks for session-based authentication (browser sessions).
        if 'user_id' in session:
            kwargs['user_id_from_token'] = session['user_id']
            return f(*args, **kwargs)

        x_auth_token = request.headers.get('x-auth-token')
        if not x_auth_token:
            # Redirects browser requests to the login page.
            if request.mimetype != 'application/json':
                return redirect(url_for('login'))
            # Returns a JSON unauthorized response for API calls.
            return jsonify({'message': 'No auth token, access denied!'}), 401

        try:
            # Decodes and verifies the JWT using the secret key.
            verified_token = jwt.decode(x_auth_token, 'password_key', algorithms=['HS256'])
            if not verified_token:
                return redirect(url_for('login')) # Redirects if token verification fails.
            
            user_id = verified_token.get('id')
            kwargs['user_id_from_token'] = user_id
            
            # Updates the session for future browser navigation.
            session['user_id'] = user_id 

            return f(*args, **kwargs)
        except jwt.PyJWTError:
            return redirect(url_for('login')) # Redirects if the token is invalid.
    return decorated_function

# ==================================
# User Registration Endpoint
# ==================================
@auth_blueprint.route('/signup', methods=['POST'])
def signup_user():
    """Handles the registration of a new user."""
    db = next(get_db())
    # Accesses the upload folder configuration.
    UPLOAD_FOLDER = current_app.config['UPLOAD_FOLDER']   
    
    try:
        data = request.form
        fullname = data.get('fullname')
        name_parts = fullname.split(maxsplit=1)
        first_name = name_parts[0] if name_parts else None
        last_name = name_parts[1] if len(name_parts) > 1 else None
        email = data.get('email')
        password = data.get('password')
        age_value = data.get('age')
        phone_value = data.get('phone') 
        
        if not email or not password or not fullname:
            return jsonify({'success': False, 'error': 'Full name, email, and password are required.'}), 400
        
        if db.query(User).filter(User.email == email).first():
            return jsonify({'success': False, 'error': 'Email already exists.'}), 409

        hashed_password = generate_password_hash(password)

        # Handles the user's profile picture upload.
        avatar_path = None
        if 'avatar' in request.files:
            avatar_file = request.files['avatar']
            if avatar_file.filename != '':
                # Generates a unique and secure filename.
                filename = f"{uuid.uuid4()}_{secure_filename(avatar_file.filename)}" 
                save_path = os.path.join(UPLOAD_FOLDER, filename) 
                avatar_file.save(save_path)
                avatar_path = filename


        # Creates the new User object and saves it to the database.
        new_user = User(
            user_id=str(uuid.uuid4()),
            email=email,
            password=hashed_password,
            first_name=first_name,
            last_name=last_name,
            phone_number=phone_value if phone_value else None, 
            age=int(age_value) if age_value and age_value.isdigit() else None,
            profile_picture_url=avatar_path,
        )
    
        db.add(new_user)
        db.commit()

        # Generates a JWT and returns a success response.
        token = jwt.encode({'user_id': new_user.user_id}, 'password_key', algorithm='HS256')       
        return jsonify({'token': token, 'message': 'User registered successfully.'}), 201

    except Exception as e:
        db.rollback()
        print(f"Registration Error: {e}")
        return jsonify({'success': False, 'error': 'An internal error occurred.'}), 500
    finally:
        db.close()


# ==================================
# User Login Endpoint
# ==================================
@auth_blueprint.route('/login', methods=['POST'])
def login_user():
    """Authenticates the user and issues a JWT."""
    db = next(get_db())
    
    if request.is_json:
        user_data = request.get_json()
    else:
        user_data = request.form.to_dict() 
    
    email = user_data.get('email')
    password = user_data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required!'}), 400
        
    # Queries the database for the user by email.
    user_db = db.query(User).filter(User.email == email).first()
    
    # Checks if the user exists.
    if not user_db:
        # Returns error if user is not found.
        return jsonify({'message': 'User with this email does not exist!'}), 400

    # Verifies the password hash.
    if not check_password_hash(user_db.password, password):
        return jsonify({'message': 'Incorrect password!'}), 400

    # Sets the session ID for browser-based authentication/redirection.
    session['user_id'] = str(user_db.user_id)
        
    token = jwt.encode({'id': user_db.user_id}, 'password_key', algorithm='HS256')
    return jsonify({
    'token': token, 
    'user': {
        'id': user_db.user_id, 
        'name': f"{user_db.first_name} {user_db.last_name}", # دمج الاسم
        'email': user_db.email
    }
}), 200
# ==================================
@auth_blueprint.route('/user/home', methods=['GET'])
@auth_middleware
def get_user_home(user_id_from_token):
    """Placeholder route for a personalized user home page."""
    return jsonify({'message': f'Welcome to your home, user {user_id_from_token}'}), 200

# ==================================
# Post Creation Endpoint
# ==================================
@auth_blueprint.route('/community/posts', methods=['POST'])
@auth_middleware
def create_post(user_id_from_token) :
    """API endpoint for creating a new post, requires authentication."""
    db = next(get_db())
    
    if request.method == 'POST':
        # Post creation logic
        try:
            data = request.get_json()
            title = data.get('title')
            body = data.get('post-body') 
            tag = data.get('tag')
            
            if not title or not body:
                return jsonify({'message': 'Title and post body are required.'}), 400

            new_post = Post(user_id=user_id_from_token, title=title, body=body, tag=tag)
            db.add(new_post)
            db.commit()
            return jsonify({'message': 'Post created successfully!', 'post_id': new_post.postID}), 201

        except Exception as e:
            db.rollback()
            return jsonify({'message': f'Server error creating post: {str(e)}'}), 500
        finally:
            db.close()
            
@auth_blueprint.route('/community/posts', methods=['GET'])
def get_community_posts():
    """API endpoint for retrieving all community posts. Does not require authentication."""
    db = next(get_db())
    try:
        # Retrieves all posts along with the creator's name, ordered by creation date.
        posts_query = db.query(Post).join(User, Post.user_id == User.user_id).order_by(Post.postDate.desc()).all()
        
        posts_list = []
        for post in posts_query:
            creator_name = "Anonymous"
            if post.creator: 
                # 1. دمج الاسم الأول والأخير
                creator_name_attempt = f"{post.creator.first_name or ''} {post.creator.last_name or ''}".strip()
                
                # 2. تعيين الاسم إذا كان موجوداً، وإلا يبقى "Anonymous"
                if creator_name_attempt:
                    creator_name = creator_name_attempt
                
            created_at_str = post.postDate.strftime("%Y-%m-%d %H:%M:%S") if post.postDate else None

            posts_list.append({
                'id': post.postID,
                'title': post.title,
                'body': post.body,
                'tag': post.tag,
                'likes_count': post.likes_count,
                'created_at': created_at_str,
                'user': {'name': creator_name}
            })
            
        return jsonify({'success': True, 'posts': posts_list}), 200

    except Exception as e:
        print(f"Failed to fetch posts: {str(e)}")
        return jsonify({'success': False, 'message': f'Failed to fetch posts: {str(e)}'}), 500
    finally:
        db.close()
# ==================================
# Uploaded Files Route
# ==================================
@auth_blueprint.route('/uploads/<filename>')
def uploaded_file(filename):
    """Route for serving uploaded files from the 'uploads' directory."""
    return send_from_directory('uploads', filename)

# ==================================
# User Profile Endpoint
# ==================================
@auth_blueprint.route('/user/profile', methods=['GET'])
@auth_middleware
def get_user_profile(user_id_from_token):
    db = next(get_db())
    user_db = db.query(User).filter(User.user_id == user_id_from_token).first()
    if not user_db:
        return jsonify({'error': 'User not found'}), 404
    
    full_name = f"{user_db.first_name or ''} {user_db.last_name or ''}".strip()
    
    return jsonify({
        'name': full_name,
        'email': user_db.email,
        # Returns the user's profile data.
        'profile_url': user_db.profile_picture_url 
    }), 200



# التعديلات الموصى بها في auth.py
@auth_blueprint.route('/update_profile', methods=['POST'])
@auth_middleware
def update_profile(user_id_from_token):
   
    db = next(get_db())
    try:
        data = request.get_json()
        
        user = db.query(User).filter(User.user_id == user_id_from_token).first()
        if not user:
            return jsonify({'success': False, 'message': 'User not found.'}), 404

        
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        
        if first_name:
           user.first_name = first_name.strip()
        if last_name:
           user.last_name = last_name.strip()
             
        user.age = data.get('age', user.age)
        user.phone_number = data.get('phone_number', user.phone_number)
        
        user.gender = data.get('gender', user.gender)
        
        new_password = data.get('password')
        if new_password:
           
            user.password = generate_password_hash(new_password)
           
        db.commit()

        return jsonify({'success': True, 'message': 'Profile updated successfully.'}), 200

    except Exception as e:
        db.rollback()
        print(f"Error updating profile for user {user_id_from_token}: {e}")
        return jsonify({'success': False, 'message': 'An internal server error occurred during update.'}), 500
    finally:
        db.close()
# ==================================
# Toggle Like Endpoint
# ==================================
@auth_blueprint.route('/toggle_like/<post_id>', methods=['POST'])
@auth_middleware
def toggle_like(post_id, user_id_from_token):
    """Handles liking/unliking a post."""
    db = next(get_db())
    try:
        post = db.query(Post).filter(Post.postID == post_id).first()
        if not post:
            return jsonify({'success': False, 'message': 'Post not found'}), 404

        # Increments the post's like count (simple counter logic without per-user tracking).
        post.likes_count = (post.likes_count or 0) + 1
        db.commit()

        new_state = True # Assuming a successful 'like' operation
        new_count = post.likes_count

        return jsonify({
            "success": True,
            "is_liked": new_state, 
            "likes_count": new_count 
        }), 200
    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        db.close()