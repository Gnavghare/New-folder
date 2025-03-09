from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField, FloatField, SelectField, SelectMultipleField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, NumberRange

class RegistrationForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class ProfileForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=100)])
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=15, max=100)])
    gender = SelectField('Gender', choices=[
        ('male', 'Male'), 
        ('female', 'Female'),
        ('other', 'Other')
    ], validators=[DataRequired()])
    height = FloatField('Height (cm)', validators=[DataRequired(), NumberRange(min=100, max=250)])
    weight = FloatField('Weight (kg)', validators=[DataRequired(), NumberRange(min=30, max=250)])
    activity_level = SelectField('Activity Level', choices=[
        ('sedentary', 'Sedentary (little or no exercise)'),
        ('lightly_active', 'Lightly Active (light exercise 1-3 days/week)'),
        ('moderately_active', 'Moderately Active (moderate exercise 3-5 days/week)'),
        ('very_active', 'Very Active (hard exercise 6-7 days/week)'),
        ('extremely_active', 'Extremely Active (very hard exercise & physical job)')
    ], validators=[DataRequired()])
    fitness_goal = SelectField('Fitness Goal', choices=[
        ('weight_loss', 'Weight Loss'),
        ('muscle_gain', 'Muscle Gain'),
        ('general_fitness', 'General Fitness')
    ], validators=[DataRequired()])
    dietary_restrictions = SelectMultipleField('Dietary Restrictions', choices=[
        ('vegetarian', 'Vegetarian'),
        ('vegan', 'Vegan'),
        ('gluten_free', 'Gluten Free'),
        ('dairy_free', 'Dairy Free'),
        ('keto', 'Keto'),
        ('paleo', 'Paleo')
    ])
    submit = SubmitField('Save Profile') 