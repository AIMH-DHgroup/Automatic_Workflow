"""Offline inventory and risk hints for bundled Triplify Java dependencies."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

KNOWN_RISK_HINTS = {
    "commons-fileupload": "1.3.1 is obsolete; review known Apache Commons FileUpload CVEs.",
    "commons-httpclient": "3.1 is end-of-life; replace with a maintained HTTP client if possible.",
    "commons-beanutils": "Old Commons BeanUtils versions have had deserialization-related CVEs.",
    "zookeeper": "3.4.x is end-of-life; review ZooKeeper CVEs before production use.",
    "log4j": "Review Log4j usage carefully; 1.x is end-of-life and old 2.x versions should be upgraded.",
    "sesame": "OpenRDF Sesame is discontinued; consider RDF4J-supported replacements.",
}


@dataclass(frozen=True)
class JarFinding:
    """One dependency inventory finding."""

    path: str
    artifact: str
    version: str
    risk: str


def parse_jar_name(path: Path) -> tuple[str, str]:
    """Parse a jar filename into artifact and version best-effort."""

    stem = path.stem
    match = re.match(r"(?P<artifact>.+)-(?P<version>\d[\w.\-]*)$", stem)
    if not match:
        return stem, "unknown"
    return match.group("artifact"), match.group("version")


def risk_hint(artifact: str) -> str:
    """Return a static risk hint for a known bundled artifact."""

    lowered = artifact.lower()
    for key, hint in KNOWN_RISK_HINTS.items():
        if key in lowered:
            return hint
    return ""


def audit_triplify_dependencies(triplify_dir: Path) -> list[JarFinding]:
    """Inventory jar dependencies and annotate known risk hints."""

    jars = sorted(triplify_dir.rglob("*.jar"))
    findings: list[JarFinding] = []
    for jar in jars:
        artifact, version = parse_jar_name(jar)
        hint = risk_hint(artifact)
        if hint:
            findings.append(
                JarFinding(
                    path=str(jar.relative_to(triplify_dir)),
                    artifact=artifact,
                    version=version,
                    risk=hint,
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the offline Triplify dependency audit."""

    parser = argparse.ArgumentParser(description="Audit bundled Triplify jar dependencies")
    parser.add_argument(
        "triplify_dir",
        nargs="?",
        type=Path,
        default=Path("M3-graph") / "Triplify",
        help="Path to the Triplify directory",
    )
    args = parser.parse_args(argv)
    findings = audit_triplify_dependencies(args.triplify_dir)
    print(json.dumps([asdict(finding) for finding in findings], indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
