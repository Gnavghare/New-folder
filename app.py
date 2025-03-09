from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import cv2
import numpy as np
import json
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from image_processor import ImageProcessor
from fitness_analyzer import FitnessAnalyzer
from plan_generator import PlanGenerator
from user_models import db, User, BodyAnalysis
from forms import RegistrationForm, LoginForm, ProfileForm

app = Flask(__name__)
app.secret_key = 'fitness_analyzer_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fitness_analyzer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize database
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize our modules
image_processor = ImageProcessor()
fitness_analyzer = FitnessAnalyzer()
plan_generator = PlanGenerator()

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if email already exists
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already registered. Please login.', 'warning')
            return redirect(url_for('login'))
        
        # Create new user
        hashed_password = generate_password_hash(form.password.data)
        new_user = User(
            name=form.name.data,
            email=form.email.data,
            password=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Login failed. Please check your email and password.', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's body analyses
    analyses = BodyAnalysis.query.filter_by(user_id=current_user.id).order_by(BodyAnalysis.created_at.desc()).all()
    return render_template('dashboard.html', analyses=analyses)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ProfileForm()
    
    # Pre-fill form with existing data if available
    if request.method == 'GET':
        form.name.data = current_user.name
        form.age.data = current_user.age
        form.gender.data = current_user.gender
        form.height.data = current_user.height
        form.weight.data = current_user.weight
        form.activity_level.data = current_user.activity_level
        form.fitness_goal.data = current_user.fitness_goal
        if current_user.dietary_restrictions:
            form.dietary_restrictions.data = current_user.dietary_restrictions.split(',')
    
    if form.validate_on_submit():
        # Update user profile
        current_user.name = form.name.data
        current_user.age = form.age.data
        current_user.gender = form.gender.data
        current_user.height = form.height.data
        current_user.weight = form.weight.data
        current_user.activity_level = form.activity_level.data
        current_user.fitness_goal = form.fitness_goal.data
        current_user.dietary_restrictions = ','.join(form.dietary_restrictions.data) if form.dietary_restrictions.data else ''
        
        db.session.commit()
        
        # Store user data in session for easy access
        session['user_data'] = current_user.to_dict()
        
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('upload'))
    
    return render_template('profile.html', form=form)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    # Check if user has completed their profile
    if not current_user.age or not current_user.height or not current_user.weight:
        flash('Please complete your profile first!', 'warning')
        return redirect(url_for('profile'))
    
    # Store user data in session if not already there
    if 'user_data' not in session:
        session['user_data'] = current_user.to_dict()
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            try:
                # Process image for body analysis
                image_rgb = image_processor.process_image(filepath)
                
                # Analyze body
                body_analysis_result = image_processor.analyze_body(image_rgb)
                
                # If body analysis was successful, analyze body status
                if body_analysis_result.get('success', False):
                    body_status = fitness_analyzer.analyze_body_status(body_analysis_result)
                    
                    # Save body analysis image
                    body_analysis_image = image_processor.get_body_analysis_image(body_analysis_result)
                    body_image_path = None
                    if body_analysis_image is not None:
                        body_image_path = filepath.replace('.', '_body_analysis.')
                        cv2.imwrite(body_image_path, body_analysis_image)
                        session['body_image_path'] = body_image_path
                    
                    # Store body analysis in session
                    session['body_analysis'] = body_status
                    
                    # Store image path in session
                    session['image_path'] = filepath
                    
                    # Save analysis to database
                    new_analysis = BodyAnalysis(
                        user_id=current_user.id,
                        original_image_path=filepath,
                        analysis_image_path=body_image_path,
                        body_type=body_status['body_measurements']['body_type'],
                        bmi_category=body_status['body_measurements']['bmi_category'],
                        body_fat_percentage=body_status['body_measurements']['body_fat_percentage'],
                        shoulder_hip_ratio=body_status['body_measurements']['shoulder_hip_ratio'],
                        posture_quality=body_status['body_measurements']['posture_quality'],
                        symmetry_score=body_status['body_measurements']['symmetry_score'],
                        analysis_data=json.dumps(body_status)
                    )
                    db.session.add(new_analysis)
                    db.session.commit()
                    
                    # Store analysis ID in session
                    session['analysis_id'] = new_analysis.id
                    
                    return redirect(url_for('results'))
                else:
                    error_message = body_analysis_result.get('message', 'No body detected in the image')
                    flash(f'Body analysis failed: {error_message}. Please upload a clear full-body image.', 'error')
                    session['body_analysis'] = None
                    return redirect(request.url)
                
            except Exception as e:
                import traceback
                traceback.print_exc()  # Print the full error traceback to the console
                flash(f'Error processing image: {str(e)}. Please try a different image.', 'error')
                return redirect(request.url)
            
    return render_template('upload.html')

@app.route('/results')
@login_required
def results():
    # Check if analysis ID is in session
    if 'analysis_id' in session:
        analysis = BodyAnalysis.query.get(session['analysis_id'])
        if analysis:
            # Load analysis data from database
            body_analysis = json.loads(analysis.analysis_data)
            user_data = current_user.to_dict()
            
            # Generate recommendations
            recommended_activities = fitness_analyzer.get_recommended_activities(user_data, body_analysis)
            exercise_plan = plan_generator.generate_exercise_plan(user_data, body_analysis)
            diet_plan = plan_generator.generate_diet_plan(user_data, body_analysis)
            
            return render_template('results.html', 
                                  user_data=user_data,
                                  body_analysis=body_analysis,
                                  recommended_activities=recommended_activities,
                                  exercise_plan=exercise_plan,
                                  diet_plan=diet_plan,
                                  image_path=analysis.original_image_path,
                                  body_image_path=analysis.analysis_image_path)
    
    # If no analysis ID or analysis not found, check session data
    if 'user_data' in session and 'body_analysis' in session:
        user_data = session['user_data']
        body_analysis = session.get('body_analysis')
        
        if not body_analysis or not body_analysis.get('success', False):
            flash('Body analysis failed. Please try uploading a different image.', 'error')
            return redirect(url_for('upload'))
        
        # Get recommended activities based on body analysis
        recommended_activities = fitness_analyzer.get_recommended_activities(user_data, body_analysis)
        
        # Generate exercise plan
        exercise_plan = plan_generator.generate_exercise_plan(user_data, body_analysis)
        
        # Generate diet plan
        diet_plan = plan_generator.generate_diet_plan(user_data, body_analysis)
        
        return render_template('results.html', 
                              user_data=user_data,
                              body_analysis=body_analysis,
                              recommended_activities=recommended_activities,
                              exercise_plan=exercise_plan,
                              diet_plan=diet_plan,
                              image_path=session.get('image_path', ''),
                              body_image_path=session.get('body_image_path', ''))
    
    # If no data available, redirect to upload
    flash('Please upload an image for analysis first.', 'warning')
    return redirect(url_for('upload'))

@app.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    analysis = BodyAnalysis.query.get_or_404(analysis_id)
    
    # Check if analysis belongs to current user
    if analysis.user_id != current_user.id:
        flash('You do not have permission to view this analysis.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Load analysis data
    body_analysis = json.loads(analysis.analysis_data)
    user_data = current_user.to_dict()
    
    # Generate recommendations
    recommended_activities = fitness_analyzer.get_recommended_activities(user_data, body_analysis)
    exercise_plan = plan_generator.generate_exercise_plan(user_data, body_analysis)
    diet_plan = plan_generator.generate_diet_plan(user_data, body_analysis)
    
    return render_template('results.html', 
                          user_data=user_data,
                          body_analysis=body_analysis,
                          recommended_activities=recommended_activities,
                          exercise_plan=exercise_plan,
                          diet_plan=diet_plan,
                          image_path=analysis.original_image_path,
                          body_image_path=analysis.analysis_image_path)

@app.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    analysis = BodyAnalysis.query.get_or_404(analysis_id)
    
    # Check if analysis belongs to current user
    if analysis.user_id != current_user.id:
        flash('You do not have permission to delete this analysis.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Delete image files
    if analysis.original_image_path and os.path.exists(analysis.original_image_path):
        os.remove(analysis.original_image_path)
    if analysis.analysis_image_path and os.path.exists(analysis.analysis_image_path):
        os.remove(analysis.analysis_image_path)
    
    # Delete analysis from database
    db.session.delete(analysis)
    db.session.commit()
    
    flash('Analysis deleted successfully.', 'success')
    return redirect(url_for('dashboard'))

# Create database tables before first request
# The before_first_request decorator is deprecated in Flask 2.x
# Using with app.app_context() instead
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True) 