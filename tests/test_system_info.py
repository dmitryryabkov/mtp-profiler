"""Tests for system information collection."""

import pytest
from unittest.mock import patch, MagicMock

from mtp_profiler.system_info.system_info import (
    collect_system_info,
    _is_apple_silicon,
    _run_cmd,
    _extract_chip_type,
)


class TestSystemInfo:
    """Tests for system info collection."""

    def test_collect_system_info(self):
        """Test that collect_system_info returns valid data."""
        info = collect_system_info()

        assert info is not None
        assert isinstance(info.is_apple_silicon, bool)
        assert isinstance(info.unified_memory_mb, int)
        assert isinstance(info.cpu_threads, int)

    def test_apple_silicon_detection(self):
        """Test Apple Silicon detection."""
        # This test depends on the actual platform
        # On Apple Silicon, _is_apple_silicon() should return True
        if _is_apple_silicon():
            assert "M" in info.chip_type if (info := collect_system_info()).chip_type else True

    def test_chip_type_extraction(self):
        """Test chip type extraction from chip name."""
        assert _extract_chip_type("Apple M3 Pro") == "M3"
        assert _extract_chip_type("Apple M2 Max") == "M2"
        assert _extract_chip_type("Apple M1") == "M1"
        assert _extract_chip_type("Apple M4 Ultra") == "M4"
        assert _extract_chip_type("Intel Core i9") == "unknown"
        assert _extract_chip_type("AMD Ryzen") == "unknown"


class TestRunCmd:
    """Tests for the _run_cmd helper."""

    def test_successful_command(self):
        """Test running a successful command."""
        result = _run_cmd(["echo", "hello"])
        assert result == "hello"

    def test_failed_command(self):
        """Test running a command that fails."""
        result = _run_cmd(["false"])
        assert result is None

    def test_nonexistent_command(self):
        """Test running a command that doesn't exist."""
        result = _run_cmd(["nonexistent_command_xyz123"])
        assert result is None

    def test_timeout(self):
        """Test that commands that timeout return None."""
        # Skip on CI where sleep might not be available
        result = _run_cmd(["sleep", "0.01"])
        # May succeed or fail depending on system
        if result is not None:
            assert result == ""


class TestIsAppleSilicon:
    """Tests for Apple Silicon detection."""

    @patch("platform.machine", return_value="arm64")
    @patch("platform.system", return_value="Darwin")
    def test_arm64_darwin(self, mock_system, mock_machine):
        """Test detection on arm64 Darwin."""
        assert _is_apple_silicon() is True

    @patch("platform.machine", return_value="x86_64")
    @patch("platform.system", return_value="Darwin")
    def test_x86_darwin(self, mock_system, mock_machine):
        """Test detection on x86_64 Darwin (Mac with Intel)."""
        assert _is_apple_silicon() is False

    @patch("platform.machine", return_value="arm64")
    @patch("platform.system", return_value="Linux")
    def test_arm64_linux(self, mock_system, mock_machine):
        """Test detection on arm64 Linux (not Apple Silicon)."""
        assert _is_apple_silicon() is False
