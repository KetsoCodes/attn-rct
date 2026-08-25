"""Find a prebuilt flash-attn wheel matching THIS environment.

Building flash-attn from source needs nvcc. The bigbatch nodes have CUDA-capable GPUs
but no CUDA compiler on PATH, so the source build fails during metadata preparation
before it even starts compiling.

Dao-AILab publishes prebuilt wheels on GitHub releases. They need no compiler, but each
is pinned to an exact combination of CUDA major, torch major.minor, C++11 ABI, CPython
version and platform. Get one wrong and you get an ImportError at runtime rather than
at install time, which is a nasty way to lose a run -- so we read all five from the live
interpreter instead of guessing from a filename.

    python find_flash_wheel.py            # print the matching URL
    python find_flash_wheel.py --install  # and pip install it
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import sysconfig
import urllib.request

RELEASES_API = "https://api.github.com/repos/Dao-AILab/flash-attention/releases"


def environment_tags():
    """Read the five tags a wheel must match from the running interpreter.

    Returns:
        Dict with torch, cuda_major, abi, python and platform.
    """
    import torch

    torch_version = ".".join(torch.__version__.split(".")[:2])       # e.g. "2.5"
    cuda_major = (torch.version.cuda or "").split(".")[0]            # e.g. "12"
    # The ABI flag must match how torch itself was compiled, or the extension fails to
    # link against libtorch at import time.
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return {
        "torch": torch_version,
        "cuda_major": cuda_major,
        "abi": abi,
        "python": python_tag,
        "platform": platform_tag,
    }


def fetch_releases(limit_pages=3):
    """List every wheel asset across the most recent releases.

    Args:
        limit_pages: pages of 100 releases to scan.

    Returns:
        (release_tag, asset_name, download_url) tuples, newest release first.
    """
    assets = []
    for page in range(1, limit_pages + 1):
        request = urllib.request.Request(
            f"{RELEASES_API}?per_page=100&page={page}",
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "attn-rct"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            page_data = json.load(response)
        if not page_data:
            break
        for release in page_data:
            for asset in release.get("assets", []):
                if asset["name"].endswith(".whl"):
                    assets.append((release["tag_name"], asset["name"],
                                   asset["browser_download_url"]))
    return assets


def match(assets, tags):
    """Filter assets down to those matching every tag.

    Args:
        assets: output of fetch_releases.
        tags: output of environment_tags.

    Returns:
        Matching assets, newest first.
    """
    needles = [
        f"cu{tags['cuda_major']}",
        f"torch{tags['torch']}",
        f"cxx11abi{tags['abi']}",
        f"{tags['python']}-{tags['python']}",
    ]
    hits = [a for a in assets if all(n in a[1] for n in needles)]
    return [a for a in hits if "linux_x86_64" in a[1]]


def report_alternatives(assets, tags):
    """Print which torch versions DO have wheels for this python and ABI.

    That is the actionable question when nothing matches: pinning torch is far cheaper
    than getting a CUDA toolkit onto a shared cluster, and the pinned version becomes
    part of the reproducibility record anyway.

    Args:
        assets: output of fetch_releases.
        tags: output of environment_tags.
    """
    combos = {}
    for _, name, _ in assets:
        if tags["python"] not in name or "linux_x86_64" not in name:
            continue
        found = re.search(r"(cu\d+)torch([\d.]+)cxx11abi(TRUE|FALSE)", name)
        if found:
            combos.setdefault(found.group(2), set()).add((found.group(1), found.group(3)))

    if not combos:
        print(f"\nNo wheels at all for {tags['python']}. Consider python 3.11 or 3.12.")
        return

    print(f"\nAvailable for {tags['python']} (linux x86_64):")
    for torch_version in sorted(combos, key=lambda v: [int(x) for x in v.split(".")]):
        variants = ", ".join(f"{c}/abi{a}" for c, a in sorted(combos[torch_version]))
        marker = "  <-- your abi" if any(
            a == tags["abi"] for _, a in combos[torch_version]) else ""
        print(f"  torch {torch_version:<6} {variants}{marker}")

    print("\nRECOMMENDED: pin torch to one of the above, e.g.")
    print("  pip install torch==<version>.* --index-url "
          "https://download.pytorch.org/whl/cu121")
    print("  python cluster/find_flash_wheel.py --install")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    tags = environment_tags()
    print("environment:")
    for key, value in tags.items():
        print(f"  {key:<10} {value}")
    print(f"  looking for: cu{tags['cuda_major']} torch{tags['torch']} "
          f"cxx11abi{tags['abi']} {tags['python']}")

    try:
        assets = fetch_releases()
    except Exception as err:  # noqa: BLE001
        print(f"FATAL: cannot reach the GitHub releases API ({err!r}).")
        print("  403 usually means an unauthenticated rate limit rather than a block;")
        print("  wait a few minutes and retry, or run this on the login node.")
        print("  Manual route: open")
        print("    https://github.com/Dao-AILab/flash-attention/releases")
        print(f"  and find an asset containing cu{tags['cuda_major']} "
              f"torch{tags['torch']} cxx11abi{tags['abi']} {tags['python']},")
        print("  then: pip install <that full URL>")
        sys.exit(1)

    print(f"scanned {len(assets)} wheels")
    hits = match(assets, tags)

    if not hits:
        print(f"\nNo wheel for torch {tags['torch']}.")
        report_alternatives(assets, tags)
        print("\nOther options if pinning is unacceptable:")
        print("  * conda install -c nvidia cuda-toolkit=12.1   (nvcc in your home dir)")
        print("  * ticket to MSS asking for CUDA toolkit on bigbatch nodes")
        sys.exit(1)

    tag, name, url = hits[0]
    print(f"\nMATCH: {name}\n  release: {tag}\n  url: {url}")

    if args.install:
        print("\ninstalling...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])
        subprocess.check_call([
            sys.executable, "-c",
            "import flash_attn; print('OK flash_attn', flash_attn.__version__)",
        ])


if __name__ == "__main__":
    main()
