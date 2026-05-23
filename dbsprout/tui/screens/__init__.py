"""TUI screens / panels mounted inside the tabbed :class:`DBSproutApp`.

Each screen is a self-contained Textual widget that reads from the shared
state DB (``.dbsprout/state.db``) — never importing CLI/generation code, in
keeping with DBSprout's read-only "visual surface" model.
"""

from __future__ import annotations
