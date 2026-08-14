#!/usr/bin/env python3
"""Present textually included GLASS headers in their public namespaces.

The implementation headers are intentionally included inside namespaces by the
umbrella headers. Doxygen parses each input file independently, so without this
adapter it records those declarations as globals and merges documentation from
otherwise distinct block/warp/thread overloads.
"""

from pathlib import Path
import re
import sys


def namespace_for(path: Path) -> tuple[str, str] | None:
    parts = path.as_posix()
    if "/src/base/" in parts:
        return "namespace glass { namespace block {", "} }"
    if "/src/cgrps/" in parts:
        return "namespace glass { namespace cgrps {", "} }"
    if "/src/nvidia/" in parts:
        if path.name == "types.cuh":
            return None  # types.cuh declares glass::nvidia itself.
        if path.name == "l1_warp.cuh":
            return "namespace glass { namespace nvidia {", "} }"
        return "namespace glass { namespace nvidia { namespace block {", "} } }"
    return None


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    path = Path(sys.argv[1]).resolve()
    source = path.read_text(encoding="utf-8")
    # All project headers are separate Doxygen inputs. Keeping their include
    # directives after adding a synthetic namespace makes Doxygen search for
    # system/vendor headers from inside that wrapper and produces irrelevant
    # include-resolution warnings. Preserve line count for source locations.
    # #pragma lines are blanked too: every header opens with `#pragma once` on
    # line 1, and the namespace wrapper must land on that line (see below).
    source = re.sub(r"(?m)^[ \t]*#(include|pragma)[^\n]*$", "", source)
    wrapper = namespace_for(path)
    if wrapper is None:
        sys.stdout.write(source)
    else:
        begin, end = wrapper
        # Open the namespace ON line 1, not above it: a leading newline would
        # shift every Doxygen source location (and thus every line number in
        # test/api-contracts.json) down by one.
        sys.stdout.write(f"{begin} {source}\n{end}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
