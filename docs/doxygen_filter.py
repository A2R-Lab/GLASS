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
    source = re.sub(r"(?m)^[ \t]*#include[^\n]*$", "", source)
    wrapper = namespace_for(path)
    if wrapper is None:
        sys.stdout.write(source)
    else:
        begin, end = wrapper
        sys.stdout.write(f"{begin}\n{source}\n{end}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
