"""MESH METHODS: leg-surface meshes (TSDF fusion) built from replayed recordings.

tsdf.py builds a mesh, holes.py closes small holes, flatten.py unrolls a saved mesh onto the
plane (SCAF / ARAP, native/ = the libigl solver), library.py saves meshes under Meshes/,
routes.py serves them to the dashboard (mounted in server.create_app).
"""

from .routes import router

__all__ = ["router"]
