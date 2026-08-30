"""Where generated figures are written.

The figure scripts are used in two places: inside the manuscript working tree,
where they write the PNGs the LaTeX source includes, and in this standalone
repository, where there is no manuscript to write into. Rather than hard-code
either, resolve it:

  1. ``ELR_FIGURE_DIR``, if set, wins. Point it at the manuscript directory to
     regenerate the figures in place.
  2. Otherwise, walk up looking for a directory containing ``main.tex`` either
     directly or in a ``manuscript/`` subdirectory. That is the manuscript
     tree.
  3. Otherwise, write to ``output/`` beside the repository root.

The third case is what a reader who clones this repository gets, and it means
the example runs without configuration.
"""

from __future__ import annotations

import os


def figure_dir(start: str) -> str:
    """Resolve the directory figures should be written to."""
    env = os.environ.get("ELR_FIGURE_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return os.path.abspath(env)

    here = os.path.abspath(start)
    probe = here
    for _ in range(4):
        probe = os.path.dirname(probe)
        if os.path.isfile(os.path.join(probe, "main.tex")):
            return probe
        sub = os.path.join(probe, "manuscript")
        if os.path.isfile(os.path.join(sub, "main.tex")):
            return sub

    root = os.path.abspath(os.path.join(here, "..", "output"))
    os.makedirs(root, exist_ok=True)
    return root
