"""RAMBO task registrations.

Imports are explicit so the external extension does not depend on
``isaaclab_tasks.utils.import_packages`` or a vendored Isaac Lab checkout.
"""

from .direct import rambo_biped, rambo_quadruped  # noqa: F401

__all__ = ["rambo_biped", "rambo_quadruped"]
