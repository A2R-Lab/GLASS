#!/usr/bin/env python3
"""Generate and check GLASS's documented public-overload inventory.

The inventory comes from Doxygen XML rather than a source regex. Coverage is
credited only when a CUDA test/example contains a compatible call shape: same
name, viable function-argument count, and viable explicit-template-argument
count. A call whose shape fits several overloads is resolved by maximum
bipartite matching — each call credits AT MOST ONE overload, so a family of
sibling overloads is only fully covered when there are enough distinctly
shaped call sites to pair off every member.

This is compile coverage, not numerical depth. Numerical, dtype,
layout, conditioning, and thread-count obligations are tracked separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT / "docs" / "doxygen" / "xml"
DEFAULT_POLICY = ROOT / "test" / "api-coverage-policy.json"
PUBLIC_TOP = {
    "glass.cuh",
    "glass-cgrps.cuh",
    "glass-defaults.cuh",
    "glass-dispatch.cuh",
    "glass-nvidia.cuh",
}


def text(node: ET.Element | None) -> str:
    return "" if node is None else "".join(node.itertext())


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


@dataclass(frozen=True)
class Contract:
    id: str
    name: str
    file: str
    line: int
    surface: str
    signature: str
    params: int
    required_params: int
    template_params: int
    minimum_explicit_template_args: int


def doxygen_version() -> str:
    out = subprocess.check_output(["doxygen", "--version"], text=True)
    return out.split()[0].strip()


def ensure_xml(xml_dir: pathlib.Path) -> None:
    if xml_dir != DEFAULT_XML:
        if not (xml_dir / "index.xml").exists():
            raise FileNotFoundError(f"no Doxygen index at {xml_dir / 'index.xml'}")
        return
    # Coverage must describe the current checkout, not whichever XML happens
    # to remain from the last local documentation build. Doxygen is fast for
    # this header-only tree, so regenerate the canonical XML every time.
    subprocess.run(["doxygen", "Doxyfile"], cwd=ROOT / "docs", check=True)


def public_location(raw: str) -> str | None:
    path = raw.replace("\\", "/")
    marker = "/GLASS/"
    if marker in path:
        path = path.split(marker, 1)[1]
    path = path.removeprefix("../")
    if path in PUBLIC_TOP:
        return path
    if path.startswith("src/internal/"):
        return None
    if path.startswith(("src/base/", "src/cgrps/", "src/nvidia/")):
        return path
    return None


_NAMESPACE_CACHE: dict[str, list[list[str]]] = {}


def namespace_by_line(rel: str) -> list[list[str]]:
    cached = _NAMESPACE_CACHE.get(rel)
    if cached is not None:
        return cached
    source = strip_comments_and_literals((ROOT / rel).read_text(errors="ignore"))
    stack: list[str | None] = []
    result: list[list[str]] = []
    token_re = re.compile(r"\bnamespace\s+([A-Za-z_]\w*)\s*\{|[{}]")
    for line in source.splitlines():
        result.append([entry for entry in stack if entry is not None])
        position = 0
        while match := token_re.search(line, position):
            token = match.group(0)
            if token == "}":
                if stack:
                    stack.pop()
            elif token == "{":
                stack.append(None)
            else:
                stack.append(match.group(1))
            position = match.end()
    _NAMESPACE_CACHE[rel] = result
    return result


def source_namespaces(rel: str, line: int) -> list[str]:
    lines = namespace_by_line(rel)
    return lines[max(0, min(line - 1, len(lines) - 1))] if lines else []


def load_contracts(xml_dir: pathlib.Path, policy: dict) -> list[Contract]:
    by_location: dict[tuple[str, int], Contract] = {}
    for xml_path in sorted(xml_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        for member in root.findall('.//memberdef[@kind="function"]'):
            if member.get("prot", "public") != "public":
                continue
            name = text(member.find("name"))
            if name.endswith("_impl") or name.startswith("_"):
                continue
            location = member.find("location")
            if location is None or not location.get("line"):
                continue
            rel = public_location(location.get("file", ""))
            if rel is None:
                continue
            if rel in policy.get("exclude_files", {}):
                continue
            line = int(location.get("line", "0"))
            namespaces = source_namespaces(rel, line)
            suffixes = policy.get("exclude_namespace_suffixes", {})
            if any(any(ns == suffix or ns.endswith(f"_{suffix}") for suffix in suffixes) for ns in namespaces):
                continue
            if f"{rel}:{name}" in policy.get("exclude_symbols", {}):
                continue
            params = member.findall("param")
            required_params = sum(p.find("defval") is None for p in params)
            template = member.find("templateparamlist")
            tparams = [] if template is None else template.findall("param")
            if rel == "src/base/dispatch.cuh":
                surface = "glass::"
            elif rel.startswith("src/base/"):
                tier = next((n for n in reversed(namespaces) if n in {"warp", "thread"}), None)
                surface = f"glass::{tier}::" if tier else "glass::block::"
            elif rel.startswith("src/cgrps/"):
                surface = "glass::cgrps::"
            elif rel.startswith("src/nvidia/"):
                tier = next((n for n in reversed(namespaces) if n in {"warp", "thread"}), None)
                surface = f"glass::nvidia::{tier}::" if tier else "glass::nvidia::block::"
            else:
                surface = "glass::"
            param_types = " ".join(text(p.find("type")) for p in params)
            minimum_explicit = 0
            for index, tparam in enumerate(tparams, start=1):
                tname = text(tparam.find("declname"))
                if not tname:
                    identifiers = re.findall(r"[A-Za-z_]\w*", text(tparam.find("type")))
                    tname = identifiers[-1] if identifiers else ""
                required = tparam.find("defval") is None
                deducible = bool(tname and re.search(rf"\b{re.escape(tname)}\b", param_types))
                if required and not deducible:
                    minimum_explicit = index
            signature = normalize(
                f"{surface}{name}{text(member.find('argsstring'))}"
            )
            digest = hashlib.sha256(f"{rel}\0{surface}\0{name}\0{signature}".encode()).hexdigest()[:12]
            contract = Contract(
                id=f"{name}-{digest}",
                name=name,
                file=rel,
                line=line,
                surface=surface,
                signature=signature,
                params=len(params),
                required_params=required_params,
                template_params=len(tparams),
                minimum_explicit_template_args=minimum_explicit,
            )
            # Doxygen sees base headers both directly and through umbrella
            # includes. The declaration's source location is the canonical
            # identity; duplicate XML views collapse here.
            by_location.setdefault((rel, line), contract)
    return sorted(by_location.values(), key=lambda c: (c.file, c.line, c.id))


def strip_comments_and_literals(source: str) -> str:
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|R\"[^\n]*?\(.*?\)[^\n]*?\"|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.S,
    )
    return pattern.sub(lambda m: "\n" * m.group(0).count("\n"), source)


def balanced_end(source: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    for pos in range(start, len(source)):
        char = source[pos]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return pos
    return None


def top_level_count(contents: str) -> int:
    if not contents.strip():
        return 0
    round_depth = square_depth = brace_depth = angle_depth = 0
    count = 1
    for char in contents:
        if char == "(": round_depth += 1
        elif char == ")": round_depth -= 1
        elif char == "[": square_depth += 1
        elif char == "]": square_depth -= 1
        elif char == "{": brace_depth += 1
        elif char == "}": brace_depth -= 1
        elif char == "<": angle_depth += 1
        elif char == ">" and angle_depth: angle_depth -= 1
        elif char == "," and not (round_depth or square_depth or brace_depth or angle_depth):
            count += 1
    return count


@dataclass(frozen=True)
class Call:
    file: str
    line: int
    name: str
    surface: str
    args: int
    explicit_template_args: int | None


def extract_calls(paths: list[pathlib.Path], api_names: set[str]) -> list[Call]:
    calls: list[Call] = []
    name_re = re.compile(r"\b((?:[A-Za-z_]\w*::)*)([a-z_]\w*)\s*")
    for path in paths:
        source = strip_comments_and_literals(path.read_text(errors="ignore"))
        aliases = {
            alias: target.rstrip(":") + "::"
            for alias, target in re.findall(
                r"\bnamespace\s+([A-Za-z_]\w*)\s*=\s*((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*;",
                source,
            )
        }
        for match in name_re.finditer(source):
            qualifier, name = match.groups()
            if qualifier:
                head, separator, tail = qualifier.partition("::")
                if separator and head in aliases:
                    qualifier = aliases[head] + tail
            if name not in api_names:
                continue
            pos = match.end()
            explicit: int | None = None
            if pos < len(source) and source[pos] == "<":
                end = balanced_end(source, pos, "<", ">")
                if end is None:
                    continue
                explicit = top_level_count(source[pos + 1 : end])
                pos = end + 1
                while pos < len(source) and source[pos].isspace():
                    pos += 1
            if pos >= len(source) or source[pos] != "(":
                continue
            end = balanced_end(source, pos, "(", ")")
            if end is None:
                continue
            calls.append(
                Call(
                    file=str(path.relative_to(ROOT)),
                    line=source.count("\n", 0, match.start()) + 1,
                    name=name,
                    surface=qualifier,
                    args=top_level_count(source[pos + 1 : end]),
                    explicit_template_args=explicit,
                )
            )
    return calls


def viable(contract: Contract, call: Call) -> bool:
    if not (contract.required_params <= call.args <= contract.params):
        return False
    explicit = 0 if call.explicit_template_args is None else call.explicit_template_args
    if not (contract.minimum_explicit_template_args <= explicit <= contract.template_params):
        return False
    if call.surface:
        if call.surface == "glass::":
            # Bare GLASS is the measured-default face of base/block declarations.
            if contract.surface not in {"glass::", "glass::block::"}:
                return False
        elif call.surface != contract.surface:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", type=pathlib.Path, default=DEFAULT_XML)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--check-manifest", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--list-gaps", action="store_true")
    parser.add_argument("--require-100", action="store_true")
    args = parser.parse_args()

    # Keep repository-relative invocations equivalent to the absolute defaults.
    # In particular, payload provenance below expects the policy path to be
    # normalized beneath ROOT before calling relative_to().
    args.xml_dir = args.xml_dir.resolve()
    args.policy = args.policy.resolve()
    if args.manifest:
        args.manifest = args.manifest.resolve()
    if args.check_manifest:
        args.check_manifest = args.check_manifest.resolve()
    if args.json:
        args.json = args.json.resolve()

    ensure_xml(args.xml_dir)
    policy = json.loads(args.policy.read_text())
    contracts = load_contracts(args.xml_dir, policy)
    evidence_paths = sorted((ROOT / "test" / "cuda").glob("*.cu"))
    evidence_paths += sorted((ROOT / "examples").glob("*.cu"))
    calls = extract_calls(evidence_paths, {c.name for c in contracts})

    by_name: dict[str, list[Contract]] = {}
    for contract in contracts:
        by_name.setdefault(contract.name, []).append(contract)
    # Maximum bipartite matching lets a constrained call disambiguate a less
    # constrained sibling. Example: cgrps axpy has both a 4-argument call and a
    # 5-argument call; the latter alone could mean an explicit group on the
    # in-place overload, but together the two shapes uniquely cover in-place
    # and out-of-place. Each call can credit at most one overload.
    edges: dict[int, list[str]] = {}
    contract_by_id = {c.id: c for c in contracts}
    ambiguous = 0
    for index, call in enumerate(calls):
        matches = [c.id for c in by_name[call.name] if viable(c, call)]
        if matches:
            edges[index] = matches
        if len(matches) > 1:
            ambiguous += 1
    matched_call: dict[str, int] = {}

    def augment(call_index: int, seen: set[str]) -> bool:
        for contract_id in sorted(edges[call_index], key=lambda cid: len(edges[matched_call[cid]]) if cid in matched_call else -1):
            if contract_id in seen:
                continue
            seen.add(contract_id)
            if contract_id not in matched_call or augment(matched_call[contract_id], seen):
                matched_call[contract_id] = call_index
                return True
        return False

    for call_index in sorted(edges, key=lambda idx: len(edges[idx])):
        augment(call_index, set())
    covered: dict[str, list[Call]] = {
        contract_id: [calls[call_index]] for contract_id, call_index in matched_call.items()
        if contract_id in contract_by_id
    }

    gaps = [c for c in contracts if c.id not in covered]
    pct = 100.0 * len(covered) / max(1, len(contracts))
    print(
        f"API overload contracts: {len(covered)}/{len(contracts)} ({pct:.1f}%); "
        f"{ambiguous} call sites required overload matching"
    )
    if args.list_gaps:
        for contract in gaps:
            print(f"  {contract.id:32s} {contract.file}:{contract.line} {contract.signature}")

    payload = {
        "schema": 1,
        "basis": "Doxygen documented public overloads",
        "doxygen_version": doxygen_version(),
        "policy": str(args.policy.relative_to(ROOT)),
        "contracts": [
            {
                **contract.__dict__,
                "evidence": [call.__dict__ for call in covered.get(contract.id, [])],
            }
            for contract in contracts
        ],
        "summary": {
            "covered": len(covered),
            "total": len(contracts),
            "percent": round(pct, 3),
            "ambiguous_calls": ambiguous,
        },
    }
    if args.manifest:
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "label": "API overloads compile-covered",
                    "message": f"{len(covered)}/{len(contracts)} ({pct:.0f}%)",
                    "color": "brightgreen" if pct == 100.0 else "red",
                }
            )
            + "\n"
        )
    stale = False
    if args.check_manifest:
        committed = (json.loads(args.check_manifest.read_text())
                     if args.check_manifest.exists() else None)
        stale = committed != payload
        if not stale:
            print("API manifest is current")
        elif committed is not None and \
                committed.get("doxygen_version") != payload["doxygen_version"]:
            # Doxygen's XML extraction differs across releases; a mismatch here
            # is toolchain skew, not necessarily a source-level drift.
            print(f"API manifest DOXYGEN VERSION SKEW: manifest was generated "
                  f"with {committed.get('doxygen_version')}, this run uses "
                  f"{payload['doxygen_version']} — align the toolchain or "
                  f"regenerate with --manifest")
        else:
            print("API manifest is stale; regenerate with --manifest")
    return 1 if (args.require_100 and gaps) or stale else 0


if __name__ == "__main__":
    sys.exit(main())
