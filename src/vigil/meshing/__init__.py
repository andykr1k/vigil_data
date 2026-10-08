"""MESH METHODS: leg-surface meshes (TSDF fusion) built from replayed recordings.

tsdf.py builds a mesh, holes.py closes small holes, library.py saves meshes under Meshes/,
routes.py serves them to the dashboard (mounted in server.create_app).
"""

from .routes import router

__all__ = ["router"]
