#!/usr/bin/env python3
"""
Interactive Curriculum Dashboard
===============================

A comprehensive Streamlit dashboard for exploring the data science curriculum,
tracking progress, and accessing all learning materials interactively.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Data Science Curriculum Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .progress-bar {
        height: 20px;
        border-radius: 10px;
        background-color: #e0e0e0;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        transition: width 0.3s ease;
    }
    .stats-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Curriculum data
CURRICULUM_DATA = {
    "modules": [
        {
            "id": 1,
            "title": "Introduction to Data Science",
            "phase": "Foundations",
            "duration": "1-2 weeks",
            "difficulty": "Beginner",
            "topics": ["Data Science Process", "CRISP-DM", "Tools Overview", "Python Basics"],
            "assessment": ["Quiz", "Exercises"],
            "status": "available"
        },
        {
            "id": 2,
            "title": "Mathematics & Statistics",
            "phase": "Foundations",
            "duration": "3-4 weeks",
            "difficulty": "Intermediate",
            "topics": ["Probability", "Distributions", "Hypothesis Testing", "Regression"],
            "assessment": ["Quiz", "8 Exercises"],
            "status": "available"
        },
        {
            "id": 3,
            "title": "Programming Foundations",
            "phase": "Foundations",
            "duration": "2-3 weeks",
            "difficulty": "Beginner",
            "topics": ["Python Syntax", "Data Structures", "NumPy", "Pandas"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 4,
            "title": "Data Collection & Storage",
            "phase": "Data Engineering",
            "duration": "1-2 weeks",
            "difficulty": "Beginner",
            "topics": ["APIs", "Databases", "ETL", "Data Formats"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 5,
            "title": "Data Cleaning & Preprocessing",
            "phase": "Data Engineering",
            "duration": "2-3 weeks",
            "difficulty": "Intermediate",
            "topics": ["Missing Data", "Outliers", "Feature Engineering", "Scaling"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 6,
            "title": "Exploratory Data Analysis",
            "phase": "Data Engineering",
            "duration": "2-3 weeks",
            "difficulty": "Intermediate",
            "topics": ["Visualization", "Statistical Analysis", "Data Storytelling"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 7,
            "title": "Machine Learning",
            "phase": "ML & Deep Learning",
            "duration": "4-5 weeks",
            "difficulty": "Advanced",
            "topics": ["Supervised Learning", "Unsupervised Learning", "Model Evaluation"],
            "assessment": ["Quiz", "Exercises"],
            "status": "available"
        },
        {
            "id": 8,
            "title": "Deep Learning",
            "phase": "ML & Deep Learning",
            "duration": "3-4 weeks",
            "difficulty": "Advanced",
            "topics": ["Neural Networks", "CNNs", "RNNs", "Transfer Learning"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 9,
            "title": "Data Visualization",
            "phase": "ML & Deep Learning",
            "duration": "2-3 weeks",
            "difficulty": "Intermediate",
            "topics": ["Advanced Plotting", "Dashboards", "Interactive Visualizations"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 10,
            "title": "Big Data Technologies",
            "phase": "Production & Career",
            "duration": "2-3 weeks",
            "difficulty": "Advanced",
            "topics": ["Spark", "Hadoop", "Distributed Computing"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 11,
            "title": "Cloud Computing",
            "phase": "Production & Career",
            "duration": "2-3 weeks",
            "difficulty": "Advanced",
            "topics": ["AWS", "GCP", "Azure", "MLOps"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 12,
            "title": "Ethics & Best Practices",
            "phase": "Production & Career",
            "duration": "1-2 weeks",
            "difficulty": "Intermediate",
            "topics": ["Responsible AI", "Bias Detection", "Data Privacy"],
            "assessment": ["Exercises"],
            "status": "available"
        },
        {
            "id": 13,
            "title": "Projects & Case Studies",
            "phase": "Production & Career",
            "duration": "3-4 weeks",
            "difficulty": "Advanced",
            "topics": ["Real-world Applications", "Portfolio Development"],
            "assessment": ["3 Projects"],
            "status": "available"
        },
        {
            "id": 14,
            "title": "Career Development",
            "phase": "Production & Career",
            "duration": "1-2 weeks",
            "difficulty": "Intermediate",
            "topics": ["Job Search", "Networking", "Certifications"],
            "assessment": ["Resources"],
            "status": "available"
        }
    ],
    "projects": [
        {
            "title": "Customer Churn Prediction",
            "type": "Predictive Analytics",
            "difficulty": "Intermediate",
            "technologies": ["Python", "scikit-learn", "pandas"],
            "description": "Build a complete customer churn prediction system"
        },
        {
            "title": "Sentiment Analysis",
            "type": "Natural Language Processing",
            "difficulty": "Advanced",
            "technologies": ["NLTK", "scikit-learn", "TensorFlow"],
            "description": "Classify movie reviews as positive or negative"
        },
        {
            "title": "Image Classification",
            "type": "Computer Vision",
            "difficulty": "Advanced",
            "technologies": ["TensorFlow", "Keras", "OpenCV"],
            "description": "Classify images using CNNs and transfer learning"
        }
    ]
}

class CurriculumDashboard:
    """Interactive curriculum dashboard"""

    def __init__(self):
        self.modules_df = pd.DataFrame(CURRICULUM_DATA["modules"])
        self.projects_df = pd.DataFrame(CURRICULUM_DATA["projects"])

        # Initialize session state for progress tracking
        if 'user_progress' not in st.session_state:
            st.session_state.user_progress = {module['id']: 0 for module in CURRICULUM_DATA["modules"]}
        if 'completed_modules' not in st.session_state:
            st.session_state.completed_modules = []
        if 'current_module' not in st.session_state:
            st.session_state.current_module = 1

    def main_dashboard(self):
        """Main dashboard view"""
        st.markdown('<h1 class="main-header">🚀 Data Science Curriculum Dashboard</h1>', unsafe_allow_html=True)

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.metric("Total Modules", len(self.modules_df))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            completed = len(st.session_state.completed_modules)
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.metric("Completed", f"{completed}/{len(self.modules_df)}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            progress_pct = int((completed / len(self.modules_df)) * 100)
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.metric("Progress", f"{progress_pct}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.metric("Projects", len(self.projects_df))
            st.markdown('</div>', unsafe_allow_html=True)

        # Progress visualization
        st.subheader("📊 Learning Progress")

        # Create progress chart
        phases = self.modules_df['phase'].unique()
        phase_progress = []

        for phase in phases:
            phase_modules = self.modules_df[self.modules_df['phase'] == phase]
            completed_in_phase = len([m for m in st.session_state.completed_modules
                                    if m in phase_modules['id'].tolist()])
            phase_progress.append({
                'Phase': phase,
                'Completed': completed_in_phase,
                'Total': len(phase_modules),
                'Percentage': (completed_in_phase / len(phase_modules)) * 100
            })

        progress_df = pd.DataFrame(phase_progress)

        fig = px.bar(progress_df, x='Phase', y='Percentage',
                    title='Progress by Learning Phase',
                    color='Percentage',
                    color_continuous_scale='Blues')
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    def curriculum_explorer(self):
        """Interactive curriculum explorer"""
        st.header("🗺️ Curriculum Explorer")

        # Phase selector
        phases = ["All"] + list(self.modules_df['phase'].unique())
        selected_phase = st.selectbox("Select Learning Phase", phases)

        # Difficulty filter
        difficulties = ["All"] + list(self.modules_df['difficulty'].unique())
        selected_difficulty = st.selectbox("Filter by Difficulty", difficulties)

        # Filter modules
        filtered_df = self.modules_df.copy()

        if selected_phase != "All":
            filtered_df = filtered_df[filtered_df['phase'] == selected_phase]

        if selected_difficulty != "All":
            filtered_df = filtered_df[filtered_df['difficulty'] == selected_difficulty]

        # Display modules
        for _, module in filtered_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**Module {module['id']}: {module['title']}**")
                    st.write(f"*{module['phase']} • {module['duration']} • {module['difficulty']}*")
                    st.write(", ".join(module['topics'][:3]) + ("..." if len(module['topics']) > 3 else ""))

                with col2:
                    progress = st.session_state.user_progress[module['id']]
                    st.progress(progress / 100)
                    st.write(f"{progress}% complete")

                with col3:
                    if st.button(f"Start Module {module['id']}", key=f"start_{module['id']}"):
                        st.session_state.current_module = module['id']
                        st.rerun()

                st.markdown("---")

    def module_details(self):
        """Detailed module view"""
        st.header("📚 Module Details")

        module_id = st.session_state.current_module
        module = self.modules_df[self.modules_df['id'] == module_id].iloc[0]

        st.markdown(f"## Module {module['id']}: {module['title']}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 Module Info")
            st.write(f"**Phase:** {module['phase']}")
            st.write(f"**Duration:** {module['duration']}")
            st.write(f"**Difficulty:** {module['difficulty']}")
            st.write(f"**Assessment:** {', '.join(module['assessment'])}")

        with col2:
            st.markdown("### 🎯 Learning Objectives")
            for topic in module['topics']:
                st.write(f"• {topic}")

        # Progress tracking
        st.markdown("### 📊 Progress Tracking")
        progress = st.slider(f"Module {module_id} Progress",
                           0, 100,
                           st.session_state.user_progress[module_id],
                           key=f"progress_{module_id}")

        if progress != st.session_state.user_progress[module_id]:
            st.session_state.user_progress[module_id] = progress
            if progress == 100 and module_id not in st.session_state.completed_modules:
                st.session_state.completed_modules.append(module_id)
                st.success(f"🎉 Module {module_id} completed!")

        # Module resources
        st.markdown("### 📁 Module Resources")

        resources = {
            "Theory": f"modules/{module_id:02d}_{module['title'].lower().replace(' ', '_').replace('&', 'and')}/README.md",
            "Code Examples": f"modules/{module_id:02d}_{module['title'].lower().replace(' ', '_').replace('&', 'and')}/examples.py",
            "Exercises": f"exercises/module_{module_id:02d}_exercises.py"
        }

        for resource_type, path in resources.items():
            if os.path.exists(path):
                st.write(f"✅ {resource_type}: Available")
                if st.button(f"Open {resource_type}", key=f"open_{resource_type}_{module_id}"):
                    st.code(open(path).read(), language='python' if path.endswith('.py') else 'markdown')
            else:
                st.write(f"❌ {resource_type}: Not available")

    def projects_showcase(self):
        """Projects showcase"""
        st.header("🚀 Projects Showcase")

        for _, project in self.projects_df.iterrows():
            with st.expander(f"**{project['title']}** - {project['type']}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Description:** {project['description']}")
                    st.write(f"**Technologies:** {', '.join(project['technologies'])}")

                with col2:
                    st.write(f"**Difficulty:** {project['difficulty']}")
                    if st.button(f"View Project", key=f"view_{project['title'].lower().replace(' ', '_')}"):
                        st.code("# Project code would be displayed here", language='python')

    def learning_analytics(self):
        """Learning analytics and insights"""
        st.header("📈 Learning Analytics")

        # Time spent learning (simulated)
        st.subheader("⏰ Time Investment")

        # Create sample time data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        time_spent = np.random.exponential(2, 30)  # Hours per day
        time_df = pd.DataFrame({'date': dates, 'hours': time_spent})

        fig = px.line(time_df, x='date', y='hours',
                     title='Daily Learning Time',
                     labels={'hours': 'Hours', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)

        # Skills progress
        st.subheader("🎯 Skills Development")

        skills = ['Python', 'Statistics', 'ML Algorithms', 'Data Viz', 'Deep Learning', 'MLOps']
        skill_levels = np.random.randint(1, 11, len(skills))

        fig = go.Figure(data=go.Scatterpolar(
            r=skill_levels,
            theta=skills,
            fill='toself'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=False,
            title="Skills Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

    def run_dashboard(self):
        """Run the complete dashboard"""
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")

        pages = {
            "🏠 Dashboard": self.main_dashboard,
            "🗺️ Curriculum Explorer": self.curriculum_explorer,
            "📚 Module Details": self.module_details,
            "🚀 Projects": self.projects_showcase,
            "📈 Analytics": self.learning_analytics
        }

        selection = st.sidebar.radio("Go to", list(pages.keys()))

        # User info sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("👤 Learning Profile")

        user_name = st.sidebar.text_input("Name", "Data Science Student")
        user_goal = st.sidebar.selectbox("Learning Goal",
                                       ["Career Transition", "Skill Enhancement", "Academic", "Personal Interest"])

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Quick Stats")
        st.sidebar.write(f"Modules Completed: {len(st.session_state.completed_modules)}")
        st.sidebar.write(f"Current Module: {st.session_state.current_module}")
        st.sidebar.write(f"Overall Progress: {int((len(st.session_state.completed_modules) / len(self.modules_df)) * 100)}%")

        # Run selected page
        pages[selection]()

        # Footer
        st.markdown("---")
        st.markdown("*Built with ❤️ for the data science community*")

def main():
    """Main dashboard application"""
    dashboard = CurriculumDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
