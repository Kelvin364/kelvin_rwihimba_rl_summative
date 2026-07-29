"""AgriScout environment package.

NOTE: this package must stay import-safe for headless training. Do NOT import
``environment.rendering`` here -- that module pulls in ``pybullet`` and is only
imported on demand when rendering is explicitly requested.
"""
