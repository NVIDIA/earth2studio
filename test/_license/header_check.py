# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""License header checker with auto-fix capability.

This script checks that all Python files have the correct SPDX license header.
It can also automatically fix files that are missing or have incorrect headers.

Usage:
    python header_check.py           # Check only (default)
    python header_check.py --fix     # Check and fix missing/incorrect headers
    python header_check.py --verbose # Show detailed output

Exit codes:
    0 - All files have correct headers (or were fixed with --fix)
    1 - Files have missing/incorrect headers (when not using --fix)
"""

import argparse
import itertools
import json
import re
import sys
from datetime import datetime
from pathlib import Path


# ANSI color codes for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def get_top_comments(data: list[str]) -> list[str]:
    """Extract comment lines from the top of a file.

    Args:
        data: List of lines from the file.

    Returns:
        List of comment lines at the top of the file.
    """
    lines_to_extract = []
    for i, line in enumerate(data):
        # Skip empty lines
        if line.strip() == "":
            continue
        # Collect comment lines
        if line.startswith("#"):
            lines_to_extract.append(i)
        # Stop at imports or docstrings
        elif "import" in line or "from" in line or '"""' in line:
            break

    return [data[line] for line in lines_to_extract]


def load_header_template(header_path: Path) -> str:
    """Load the header template file.

    Args:
        header_path: Path to the header template file.

    Returns:
        Header template as a string.
    """
    with open(header_path, encoding="utf-8") as f:
        return f.read()


def check_file_header(
    filepath: Path,
    header_template: str,
    starting_year: int,
    current_year: int,
) -> dict:
    """Check if a file has a valid license header.

    Args:
        filepath: Path to the file to check.
        header_template: Expected header template.
        starting_year: First valid year for copyright.
        current_year: Current year.

    Returns:
        Dict with keys:
            - valid: bool, whether header is valid
            - error: str or None, error message if invalid
            - has_gpl: bool, whether file contains GPL reference
            - existing_header: str, the current header (if any)
    """
    result = {
        "valid": False,
        "error": None,
        "has_gpl": False,
        "existing_header": "",
    }

    try:
        with open(filepath, encoding="utf-8") as f:
            data = f.readlines()
    except Exception as e:
        result["error"] = f"Failed to read file: {e}"
        return result

    comments = get_top_comments(data)
    result["existing_header"] = "".join(comments)

    # Check for ignore directive
    if comments and "# ignore_header_test" in comments[0]:
        result["valid"] = True
        return result

    # Check for GPL license (not allowed)
    for line in comments:
        if "gpl" in line.lower():
            result["has_gpl"] = True
            result["error"] = "File contains GPL license reference (not allowed)"
            return result

    # Check for NVIDIA copyright
    found_nvidia = False
    year_valid = False

    for line in comments:
        if re.search(r"Copyright.*NVIDIA", line, re.IGNORECASE):
            found_nvidia = True
            # Check year is valid
            for year in range(starting_year, current_year + 1):
                if str(year) in line:
                    year_valid = True
                    break

    if not found_nvidia:
        result["error"] = "Missing NVIDIA copyright line"
        return result

    if not year_valid:
        result["error"] = (
            f"Copyright year not in valid range ({starting_year}-{current_year})"
        )
        return result

    # Check for SPDX identifier
    has_spdx = any("SPDX-License-Identifier" in line for line in comments)
    if not has_spdx:
        result["error"] = "Missing SPDX-License-Identifier line"
        return result

    result["valid"] = True
    return result


def fix_file_header(filepath: Path, header_template: str) -> bool:
    """Add or replace the license header in a file.

    Args:
        filepath: Path to the file to fix.
        header_template: Header template to use.

    Returns:
        True if file was modified, False otherwise.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return False

    lines = content.split("\n")

    # Find where the actual code starts (skip existing header comments)
    code_start = 0
    in_header = True
    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_header:
            if stripped == "" or stripped.startswith("#"):
                continue
            else:
                code_start = i
                break

    # Check if there's a shebang line to preserve
    shebang = ""
    if lines and lines[0].startswith("#!"):
        shebang = lines[0] + "\n"
        if code_start == 0:
            code_start = 1

    # Build new content
    remaining_content = "\n".join(lines[code_start:])

    # Ensure header ends with newline and code starts with blank line
    new_content = (
        shebang + header_template.rstrip() + "\n\n" + remaining_content.lstrip()
    )

    # Write back
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check and optionally fix license headers in Python files."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix files with missing or incorrect headers",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for each file",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    args = parser.parse_args()

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.RED = ""
        Colors.GREEN = ""
        Colors.YELLOW = ""
        Colors.BLUE = ""
        Colors.RESET = ""
        Colors.BOLD = ""

    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    header_path = Path(__file__).parent / config["copyright_file"]
    working_path = (Path(__file__).parent / config["dir"]).resolve()
    exts = config["include-ext"]
    exclude_patterns = config.get("exclude-patterns", [])

    header_template = load_header_template(header_path)
    current_year = datetime.today().year
    starting_year = 2023

    # Build exclude set
    exclude_paths = [
        (Path(__file__).parent / Path(path)).resolve().rglob("*")
        for path in config["exclude-dir"]
    ]
    all_exclude_paths = itertools.chain.from_iterable(exclude_paths)
    exclude_filenames = {p for p in all_exclude_paths if p.suffix in exts}

    # Find all files to check
    filenames = [
        p
        for p in working_path.rglob("*")
        if p.suffix in exts
        and p not in exclude_filenames
        and not any(pattern in p.parts for pattern in exclude_patterns)
    ]

    # Track results
    valid_files = []
    problematic_files = []
    fixed_files = []
    gpl_files = []

    print(f"\n{Colors.BOLD}License Header Check{Colors.RESET}")
    print(f"{'=' * 50}")
    print(f"Checking {len(filenames)} Python files...\n")

    for filepath in filenames:
        result = check_file_header(
            filepath, header_template, starting_year, current_year
        )

        if result["valid"]:
            valid_files.append(filepath)
            if args.verbose:
                print(f"{Colors.GREEN}[OK]{Colors.RESET} {filepath}")
        elif result["has_gpl"]:
            gpl_files.append((filepath, result["error"]))
            print(f"{Colors.RED}[GPL]{Colors.RESET} {filepath}")
            print(f"       Error: {result['error']}")
        else:
            if args.fix:
                if fix_file_header(filepath, header_template):
                    fixed_files.append(filepath)
                    print(f"{Colors.YELLOW}[FIXED]{Colors.RESET} {filepath}")
                else:
                    problematic_files.append((filepath, result["error"]))
                    print(f"{Colors.RED}[FAIL]{Colors.RESET} {filepath}")
                    print(f"       Error: {result['error']}")
            else:
                problematic_files.append((filepath, result["error"]))
                print(f"{Colors.RED}[FAIL]{Colors.RESET} {filepath}")
                print(f"       Error: {result['error']}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"{Colors.BOLD}Summary{Colors.RESET}")
    print(f"{'=' * 50}")
    print(f"  Total files checked: {len(filenames)}")
    print(f"  {Colors.GREEN}Valid:{Colors.RESET} {len(valid_files)}")
    if fixed_files:
        print(f"  {Colors.YELLOW}Fixed:{Colors.RESET} {len(fixed_files)}")
    if problematic_files:
        print(f"  {Colors.RED}Invalid:{Colors.RESET} {len(problematic_files)}")
    if gpl_files:
        print(f"  {Colors.RED}GPL (blocked):{Colors.RESET} {len(gpl_files)}")

    # Provide actionable fix instructions for agents
    if problematic_files and not args.fix:
        print(f"\n{Colors.BOLD}How to Fix{Colors.RESET}")
        print(f"{'=' * 50}")
        print("Option 1: Run with --fix flag to auto-fix all files:")
        print(f"  {Colors.BLUE}python {__file__} --fix{Colors.RESET}")
        print()
        print("Option 2: Manually add this header to the top of each file:")
        print(f"{Colors.YELLOW}")
        print(header_template)
        print(f"{Colors.RESET}")
        print("Files needing fixes:")
        for filepath, error in problematic_files:
            rel_path = (
                filepath.relative_to(working_path)
                if filepath.is_relative_to(working_path)
                else filepath
            )
            print(f"  - {rel_path}: {error}")

    if gpl_files:
        print(f"\n{Colors.RED}GPL License Detected (Manual Fix Required){Colors.RESET}")
        print("The following files contain GPL references which are not allowed:")
        for filepath, error in gpl_files:
            print(f"  - {filepath}")
        print("Please remove GPL references and use Apache-2.0 license instead.")

    # Exit code
    if problematic_files or gpl_files:
        if args.fix and not gpl_files:
            # All fixable issues were fixed
            print(
                f"\n{Colors.GREEN}All fixable issues have been resolved.{Colors.RESET}"
            )
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}All files have valid license headers!{Colors.RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
