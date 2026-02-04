#!/usr/bin/env python3
"""
Post-process sphinx-apidoc output.

Goals
-----
1) Reorganize module/package .rst files into folders mirroring the Python
   package structure:
      mypkg.sub.mod.rst  ->  mypkg/sub/mod.rst
2) Shorten the top header in each .rst by removing the fully-qualified prefix:
      "mypkg.sub.mod.MyClass" -> "MyClass"
      "mypkg.sub.mod"         -> "mod"   (configurable: see --header-mode)
3) Update all toctrees / TOC entries to point to the new paths.

Usage
-----
  python post_apidoc.py --apidoc-out docs/api --package mypkg

Options
-------
  --header-mode last      : header becomes last segment only (default)
  --header-mode drop-root : drop just "mypkg." prefix, keep rest (e.g. sub.mod)

Notes
-----
- Updates toctree entries conservatively: replaces entries that exactly match an
  old docname (either "mypkg.sub.mod" or "mypkg.sub.mod.rst") with the new
  docname ("mypkg/sub/mod").
- Also updates common :doc:`...` roles and simple references of that form.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

HDR_UNDERLINE_CHARS = r"=~\-^\"'#*+`:.><_"


def iter_rst_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.rst")


def is_probably_header(lines: List[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    title = lines[i].rstrip("\n")
    underline = lines[i + 1].rstrip("\n")
    if not title.strip() or not underline.strip():
        return False
    if len(underline) < len(title):
        return False
    if len(set(underline)) != 1:
        return False
    if underline[0] not in HDR_UNDERLINE_CHARS:
        return False
    return True


def shorten_title(title: str, package: str, mode: str) -> str:
    t = title.strip()
    if mode == "drop-root":
        if t.startswith(package + "."):
            return t[len(package) + 1 :]
        return t
    # mode == "last"
    if t.startswith(package + "."):
        t = t[len(package) + 1 :]
    return t.split(".")[-1] if "." in t else t


def rewrite_top_header(text: str, package: str, mode: str) -> str:
    lines = text.splitlines(True)
    for i in range(len(lines) - 1):
        if is_probably_header(lines, i):
            title = lines[i].rstrip("\n")
            underline = lines[i + 1].rstrip("\n")
            new_title = shorten_title(title, package, mode)
            if new_title != title:
                ch = underline[0]
                lines[i] = new_title + "\n"
                lines[i + 1] = (ch * max(1, len(new_title))) + "\n"
            break
    return "".join(lines)


def guess_module_name_from_file(stem: str, package: str) -> Optional[str]:
    # Common apidoc "meta" files
    if stem in {"modules", "packages", "index"}:
        return None
    # Common apidoc convention: dotted full module path as stem
    if stem == package or stem.startswith(package + "."):
        return stem
    return None


def module_to_relpath(module: str) -> Path:
    # mypkg.sub.mod -> mypkg/sub/mod.rst
    return Path(*module.split(".")).with_suffix(".rst")


def posix_docname(rel_rst: Path) -> str:
    return rel_rst.with_suffix("").as_posix()


@dataclass
class MovePlan:
    src: Path
    dst: Path
    module: str


def build_move_plans(outdir: Path, package: str) -> List[MovePlan]:
    plans: List[MovePlan] = []
    for p in iter_rst_files(outdir):
        mod = guess_module_name_from_file(p.stem, package)
        if not mod:
            continue
        dst = outdir / module_to_relpath(mod)
        if p.resolve() == dst.resolve():
            continue
        plans.append(MovePlan(src=p, dst=dst, module=mod))
    return plans


def apply_docname_rewrites(text: str, doc_map: Dict[str, str]) -> str:
    """
    Rewrite doc references in:
      - toctree entries (plain lines)
      - :doc:`...` roles
      - .. include:: ... (optional but helpful)

    We rewrite only exact old docnames (with either dot form or old path form),
    optionally with ".rst" suffix.
    """

    # 1) Rewrite :doc:`...` roles
    def repl_doc_role(m: re.Match) -> str:
        target = m.group(1).strip()
        # handle :doc:`Title <target>` form
        if "<" in target and target.endswith(">"):
            # split on last '<' to tolerate titles containing '<'
            left, right = target.rsplit("<", 1)
            t = right[:-1].strip()
            t_norm = t[:-4] if t.endswith(".rst") else t
            if t_norm in doc_map:
                t_new = doc_map[t_norm]
                return f":doc:`{left}<{t_new}>`"
            return m.group(0)

        t = target
        t_norm = t[:-4] if t.endswith(".rst") else t
        if t_norm in doc_map:
            return f":doc:`{doc_map[t_norm]}`"
        return m.group(0)

    text = re.sub(r":doc:`([^`]+)`", repl_doc_role, text)

    # 2) Rewrite include directives (absolute or relative)
    def repl_include(m: re.Match) -> str:
        indent, path = m.group(1), m.group(2).strip()
        # normalize
        p = path.replace("\\", "/")
        # drop leading /
        if p.startswith("/"):
            p0 = p[1:]
        else:
            p0 = p
        p0 = p0[:-4] if p0.endswith(".rst") else p0
        if p0 in doc_map:
            # keep original leading slash style if it had one
            newp = doc_map[p0]
            if p.startswith("/"):
                newp = "/" + newp
            return f"{indent}.. include:: {newp}.rst\n"
        return m.group(0)

    text = re.sub(r"(?m)^(\s*)\.\.\s*include::\s*([^\n]+)\n", repl_include, text)

    # 3) Rewrite toctree entry lines.
    # We do this line-based to preserve indentation and avoid touching directives.
    lines = text.splitlines(True)
    out: List[str] = []
    in_toctree = False
    toctree_indent: Optional[int] = None

    for line in lines:
        raw = line.rstrip("\n")
        stripped = raw.strip()

        # Detect start of toctree directive
        if stripped.startswith(".. toctree::"):
            in_toctree = True
            toctree_indent = len(raw) - len(raw.lstrip(" "))
            out.append(line)
            continue

        if in_toctree:
            # End of directive block when indentation returns to <= toctree indent
            # AND it's not a blank line.
            cur_indent = len(raw) - len(raw.lstrip(" "))
            if stripped and toctree_indent is not None and cur_indent <= toctree_indent:
                in_toctree = False
                toctree_indent = None
                # fall through and process line normally (outside toctree)

            else:
                # Inside toctree: skip option lines like ":maxdepth:"
                if not stripped or stripped.startswith(":"):
                    out.append(line)
                    continue

                # A toctree entry: may include glob patterns; we won't rewrite globs.
                # Only rewrite plain docnames/paths.
                entry = stripped.replace("\\", "/")
                entry_norm = entry[:-4] if entry.endswith(".rst") else entry

                if entry_norm in doc_map:
                    new_entry = doc_map[entry_norm]
                    indent = raw[: len(raw) - len(raw.lstrip(" "))]
                    out.append(f"{indent}{new_entry}\n")
                else:
                    out.append(line)
                continue

        # Outside toctree: leave untouched
        out.append(line)

    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apidoc-out", required=True, type=Path)
    ap.add_argument("--package", required=True, help="e.g. mypkg")
    ap.add_argument(
        "--header-mode",
        choices=["last", "drop-root"],
        default="last",
        help="How to shorten the top header title",
    )
    ap.add_argument("--no-redirects", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    outdir: Path = args.apidoc_out
    package: str = args.package
    make_redirects = not args.no_redirects

    if not outdir.exists():
        raise SystemExit(f"Output directory does not exist: {outdir}")

    plans = build_move_plans(outdir, package)

    # Map old docname -> new docname (relative to outdir, no suffix)
    doc_map: Dict[str, str] = {}
    for plan in plans:
        old_rel = plan.src.relative_to(outdir)
        new_rel = plan.dst.relative_to(outdir)
        doc_map[posix_docname(old_rel)] = posix_docname(new_rel)

    # Also map dotted docnames "mypkg.sub.mod" (common in apidoc toctrees)
    for plan in plans:
        dotted = plan.module
        new_rel = plan.dst.relative_to(outdir)
        doc_map[dotted] = posix_docname(new_rel)

    if args.dry_run:
        print(f"Would move {len(plans)} files.")
        for plan in plans[:50]:
            print(f"  {plan.src.relative_to(outdir)} -> {plan.dst.relative_to(outdir)}")
        if len(plans) > 50:
            print(f"  ... ({len(plans)-50} more)")
        print("Would rewrite toctrees/doc refs and headers.")
        return

    # A) Rewrite headers in all .rst (before moving)
    for p in list(iter_rst_files(outdir)):
        txt = p.read_text(encoding="utf-8")
        new_txt = rewrite_top_header(txt, package, args.header_mode)
        if new_txt != txt:
            p.write_text(new_txt, encoding="utf-8")

    # B) Move files and optionally create stubs at old paths
    for plan in plans:
        plan.dst.parent.mkdir(parents=True, exist_ok=True)
        if plan.dst.exists():
            plan.dst.unlink()
        shutil.move(str(plan.src), str(plan.dst))

        if make_redirects:
            # Keep old docname alive: old file includes the new one
            plan.src.parent.mkdir(parents=True, exist_ok=True)
            stub = f".. include:: /{posix_docname(plan.dst.relative_to(outdir))}.rst\n"
            plan.src.write_text(stub, encoding="utf-8")

    # C) Update toctrees + doc refs everywhere (THIS fixes TOCs for new folders)
    for p in list(iter_rst_files(outdir)):
        txt = p.read_text(encoding="utf-8")
        new_txt = apply_docname_rewrites(txt, doc_map)
        if new_txt != txt:
            p.write_text(new_txt, encoding="utf-8")

    print("Done.")
    print(f"Moved {len(plans)} files and updated toctrees/doc refs.")


if __name__ == "__main__":
    main()
