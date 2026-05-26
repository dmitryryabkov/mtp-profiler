"""System information collection for Apple Silicon Macs.

Collects hardware and OS metadata relevant to MTP profiling:
- Chip type (M1, M2, M3, etc.)
- Unified memory size
- macOS version
- CPU thread count
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """Apple Silicon system information."""

    chip: str = ""
    chip_type: str = ""  # M1, M2, M3, etc.
    unified_memory_mb: int = 0
    macos_version: str = ""
    cpu_threads: int = 0
    cpu_total_threads: int = 0
    is_apple_silicon: bool = False


def collect_system_info() -> SystemInfo:
    """Collect system information from the local machine.

    Uses sysctl, system_profiler, and platform module.
    Gracefully handles missing tools or unexpected output.

    Returns:
        SystemInfo with collected data.
    """
    info = SystemInfo()

    # Check if Apple Silicon
    info.is_apple_silicon = _is_apple_silicon()

    if info.is_apple_silicon:
        info.chip = _get_chip_name()
        info.chip_type = _extract_chip_type(info.chip)
        info.unified_memory_mb = _get_unified_memory_mb()
        info.macos_version = _get_macos_version()
        info.cpu_threads = _get_cpu_threads()
        info.cpu_total_threads = _get_total_threads()
    else:
        # Non-Apple Silicon: collect what we can
        info.macos_version = _get_macos_version()
        info.cpu_threads = _get_cpu_threads()
        info.cpu_total_threads = _get_total_threads()
        info.chip = platform.machine()

    return info


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.machine() in ("arm64",) and platform.system() == "Darwin"


def _run_cmd(cmd: list[str]) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _get_chip_name() -> str:
    """Get the chip name via sysctl."""
    output = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    if output:
        return output

    # Try system_profiler
    output = _run_cmd(["system_profiler", "SPHardwareDataType"])
    if output and "Chip" in output:
        for line in output.split("\n"):
            if "Chip" in line:
                # "Chip: M3 Pro"
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[1].strip()

    return platform.machine()


def _extract_chip_type(chip_name: str) -> str:
    """Extract M1/M2/M3/etc from chip name."""
    import re

    match = re.search(r"(M\d+)", chip_name)
    if match:
        return match.group(1)
    return "unknown"


def _get_unified_memory_mb() -> int:
    """Get unified memory size in MB."""
    # Try sysctl first
    output = _run_cmd(["sysctl", "-n", "hw.memsize"])
    if output:
        try:
            return int(output) // (1024 * 1024)
        except ValueError:
            pass

    # Try system_profiler
    output = _run_cmd(["system_profiler", "SPHardwareDataType"])
    if output:
        for line in output.split("\n"):
            line = line.strip()
            if "Memory" in line:
                # "Memory: 18 GB" or "Unified Memory: 36 GB"
                import re
                match = re.search(r"(\d+)", line)
                if match:
                    gb = int(match.group(1))
                    return gb * 1024

    return 0


def _get_macos_version() -> str:
    """Get macOS version string."""
    output = _run_cmd(["sw_vers", "-productVersion"])
    if output:
        return output

    return platform.release()


def _get_cpu_threads() -> int:
    """Get number of available CPU threads."""
    try:
        return int(_run_cmd(["sysctl", "-n", "hw.logicalcpu"]) or "0")
    except (ValueError, TypeError):
        return 0


def _get_total_threads() -> int:
    """Get total CPU count (including performance cores)."""
    try:
        return int(_run_cmd(["sysctl", "-n", "hw.ncpu"]) or "0")
    except (ValueError, TypeError):
        return 0
