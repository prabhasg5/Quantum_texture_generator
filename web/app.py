from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import secrets
import io
import base64
from pathlib import Path

# Import texture generator utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from web.generation_utils import TextureGenerator

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///quantum_canvas.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

db = SQLAlchemy(app)

# ===== DATABASE MODELS =====
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    projects = db.relationship('Project', backref='owner', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


class Project(db.Model):
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    project_type = db.Column(db.String(50), default='canvas')  # 'canvas' or 'generation'
    canvas_data = db.Column(db.JSON)  # Store canvas state
    thumbnail = db.Column(db.Text)  # Base64 thumbnail image
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    generations = db.relationship('Generation', backref='project', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'project_type': self.project_type,
            'canvas_data': self.canvas_data,
            'thumbnail': self.thumbnail,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'generation_count': len(self.generations)
        }


class Generation(db.Model):
    __tablename__ = 'generations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=True)  # Optional
    texture_class = db.Column(db.String(50))
    num_samples = db.Column(db.Integer, default=6)
    image_data = db.Column(db.Text)  # Store base64 image data
    file_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    generation_metadata = db.Column(db.JSON)
    
    def to_dict(self):
        return {
            'id': self.id,
            'texture_class': self.texture_class,
            'num_samples': self.num_samples,
            'image_data': self.image_data,
            'file_path': self.file_path,
            'created_at': self.created_at.isoformat(),
            'generation_metadata': self.generation_metadata
        }


# ===== ROUTES =====
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/api/signup', methods=['POST'])
def signup():
    """User registration"""
    try:
        data = request.get_json()
        
        # Validate input
        if not all(k in data for k in ['name', 'email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=data['email'].lower()).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Validate password strength
        if len(data['password']) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        # Create new user
        user = User(
            name=data['name'],
            email=data['email'].lower()
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        # Auto sign-in by creating session
        session['user_id'] = user.id
        
        return jsonify({
            'message': 'Account created successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/signin', methods=['POST'])
def signin():
    """User authentication"""
    try:
        data = request.get_json()
        
        if not all(k in data for k in ['email', 'password']):
            return jsonify({'error': 'Missing email or password'}), 400
        
        user = User.query.filter_by(email=data['email'].lower()).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is inactive'}), 403
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Create session
        session.permanent = True
        session['user_id'] = user.id
        session['user_email'] = user.email
        session['user_name'] = user.name
        
        return jsonify({
            'message': 'Sign in successful',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/signout', methods=['POST'])
def signout():
    """User logout"""
    session.clear()
    return jsonify({'message': 'Signed out successfully'}), 200


@app.route('/dashboard')
def dashboard():
    """User dashboard - texture generation interface"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('index'))
    
    session['user_name'] = user.name
    return render_template('dashboard.html', user=user)


@app.route('/canvas')
def canvas():
    """Canvas/Moodboard page"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('index'))
    
    return render_template('canvas.html', user=user)


@app.route('/api/texture-classes')
def get_texture_classes():
    """Get available texture classes"""
    try:
        generator = get_texture_generator()
        classes = generator.get_available_classes()
        return jsonify({'classes': classes}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate_texture():
    """Generate textures with color palette and concepts"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        texture_class = data.get('texture_class')
        num_samples = data.get('num_samples', 6)
        
        if not texture_class:
            return jsonify({'error': 'Texture class required'}), 400
        
        if num_samples < 1 or num_samples > 12:
            return jsonify({'error': 'num_samples must be between 1 and 12'}), 400
        
        # Generate textures
        generator = get_texture_generator()
        grid_image, color_palette, concept_words = generator.generate_textures(
            texture_class, num_samples
        )
        
        # Convert image to base64
        buffered = io.BytesIO()
        grid_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save generation to database
        user = User.query.get(session['user_id'])
        
        # Create generation without project (independent)
        generation = Generation(
            user_id=user.id,
            project_id=None,  # Not associated with any project initially
            texture_class=texture_class,
            num_samples=num_samples,
            image_data=f'data:image/png;base64,{img_str}',  # Store the full base64 image
            generation_metadata={
                'color_palette': color_palette,
                'concept_words': concept_words
            }
        )
        db.session.add(generation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'color_palette': color_palette,
            'concept_words': concept_words,
            'generation_id': generation.id
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@app.route('/api/generations/history')
def get_generation_history():
    """Get user's generation history"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user = User.query.get(session['user_id'])
        
        # Get all generations for the user, sorted by most recent
        generations = Generation.query.filter_by(user_id=user.id)\
            .order_by(Generation.created_at.desc())\
            .limit(50)\
            .all()
        
        return jsonify({
            'generations': [gen.to_dict() for gen in generations]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/save', methods=['POST'])
def save_canvas_project():
    """Save or update a canvas project"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        project_id = data.get('project_id')
        project_name = data.get('name', 'Untitled Canvas')
        canvas_data = data.get('canvas_data', {})
        thumbnail = data.get('thumbnail')  # base64 image
        
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if project_id:
            # Update existing project
            project = Project.query.filter_by(id=project_id, user_id=user.id).first()
            if not project:
                return jsonify({'error': 'Project not found'}), 404
            
            project.name = project_name
            project.canvas_data = canvas_data
            project.thumbnail = thumbnail
            project.updated_at = datetime.utcnow()
        else:
            # Create new project
            project = Project(
                user_id=user.id,
                name=project_name,
                project_type='canvas',
                canvas_data=canvas_data,
                thumbnail=thumbnail
            )
            db.session.add(project)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'project_id': project.id,
            'message': 'Canvas saved successfully'
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/list', methods=['GET'])
def list_canvas_projects():
    """Get all canvas projects for the current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        projects = Project.query.filter_by(
            user_id=user.id,
            project_type='canvas'
        ).order_by(Project.updated_at.desc()).all()
        
        projects_data = []
        for project in projects:
            projects_data.append({
                'id': project.id,
                'name': project.name,
                'description': project.description,
                'thumbnail': project.thumbnail,
                'created_at': project.created_at.isoformat(),
                'updated_at': project.updated_at.isoformat(),
                'item_count': len(project.canvas_data.get('items', [])) if project.canvas_data else 0
            })
        
        return jsonify({'projects': projects_data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/load/<int:project_id>', methods=['GET'])
def load_canvas_project(project_id):
    """Load a specific canvas project"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        project = Project.query.filter_by(id=project_id, user_id=user.id).first()
        if not project:
            return jsonify({'error': 'Project not found'}), 404
        
        return jsonify({
            'project': {
                'id': project.id,
                'name': project.name,
                'description': project.description,
                'canvas_data': project.canvas_data or {},
                'created_at': project.created_at.isoformat(),
                'updated_at': project.updated_at.isoformat()
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/delete/<int:project_id>', methods=['DELETE'])
def delete_canvas_project(project_id):
    """Delete a canvas project"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        project = Project.query.filter_by(id=project_id, user_id=user.id).first()
        if not project:
            return jsonify({'error': 'Project not found'}), 404
        
        db.session.delete(project)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Project deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/old-dashboard')
def old_dashboard():
    """Old placeholder dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('index'))
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard - Quantum Canvas</title>
        <style>
            body {{
                font-family: 'Space Grotesk', sans-serif;
                background: #0a0a0f;
                color: white;
                padding: 2rem;
                text-align: center;
            }}
            h1 {{
                background: linear-gradient(135deg, #00f5ff, #ff00ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 3rem;
                margin-bottom: 1rem;
            }}
            .btn {{
                padding: 1rem 2rem;
                background: linear-gradient(135deg, #00f5ff, #ff00ff);
                border: none;
                border-radius: 50px;
                color: #0a0a0f;
                font-weight: 700;
                cursor: pointer;
                margin: 1rem;
                text-decoration: none;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <h1>Welcome, {user.name}!</h1>
        <p>Dashboard coming soon...</p>
        <button class="btn" onclick="window.location.href='/api/signout'">Sign Out</button>
    </body>
    </html>
    """


@app.route('/api/user/profile')
def get_profile():
    """Get current user profile"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify(user.to_dict()), 200


# ===== DATABASE INITIALIZATION =====
# Initialize texture generator (lazy loading)
_texture_generator = None

def get_texture_generator():
    """Get or initialize texture generator."""
    global _texture_generator
    if _texture_generator is None:
        # Initialize without checkpoint - will load from dataset
        _texture_generator = TextureGenerator()
    return _texture_generator


def init_db():
    """Initialize database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")


# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
