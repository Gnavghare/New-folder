from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    activity_level = db.Column(db.String(50))
    fitness_goal = db.Column(db.String(50))
    dietary_restrictions = db.Column(db.String(200))
    analyses = db.relationship('BodyAnalysis', backref='user', lazy=True)

    @property
    def has_completed_profile(self):
        """Check if user has completed their profile with required information."""
        return all([
            self.age is not None,
            self.gender is not None,
            self.height is not None,
            self.weight is not None,
            self.activity_level is not None,
            self.fitness_goal is not None
        ])

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'age': self.age,
            'gender': self.gender,
            'height': self.height,
            'weight': self.weight,
            'activity_level': self.activity_level,
            'fitness_goal': self.fitness_goal,
            'dietary_restrictions': self.dietary_restrictions
        }

class BodyAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    original_image_path = db.Column(db.String(200))
    analysis_image_path = db.Column(db.String(200))
    body_type = db.Column(db.String(50))
    bmi_category = db.Column(db.String(50))
    body_fat_percentage = db.Column(db.Float)
    shoulder_hip_ratio = db.Column(db.Float)
    posture_quality = db.Column(db.String(50))
    symmetry_score = db.Column(db.Float)
    analysis_data = db.Column(db.Text) 