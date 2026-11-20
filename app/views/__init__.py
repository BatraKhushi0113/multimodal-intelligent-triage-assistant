"""
View modules for Multimodal Intelligent Triage Assistant.
"""

from app.views.architecture_view import render_architecture_view
from app.views.assessment_view import render_assessment_view
from app.views.benchmarks_view import render_benchmarks_view
from app.views.intake_view import render_intake_view
from app.views.landing_view import render_landing_view
from app.views.profile_view import render_profile_view
from app.views.radiology_view import render_radiology_view

__all__ = [
    "render_architecture_view",
    "render_assessment_view",
    "render_benchmarks_view",
    "render_intake_view",
    "render_landing_view",
    "render_profile_view",
    "render_radiology_view",
]

