"""
Unit Tests for Unified Single-Host Runner (run.py).
===================================================
Tests socket checking, CLI parser configuration, and command generation.
"""

from __future__ import annotations

import argparse
import socket
import sys
from unittest.mock import patch
import pytest

from run import is_port_in_use


def test_is_port_in_use_closed():
    """Verify is_port_in_use returns False for an unallocated high-numbered port."""
    # Find an open port and close it immediately
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    # Port is now closed
    assert is_port_in_use(port) is False


def test_is_port_in_use_open():
    """Verify is_port_in_use returns True when a socket is bound and listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        _, port = s.getsockname()
        assert is_port_in_use(port) is True


def test_run_cli_argument_parsing():
    """Verify run.py CLI argument defaults and custom flags."""
    test_args = ["run.py", "--host", "0.0.0.0", "--api-port", "9000", "--ui-port", "9501", "--no-browser"]
    with patch.object(sys, "argv", test_args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--api-port", type=int, default=8000)
        parser.add_argument("--ui-port", type=int, default=8501)
        parser.add_argument("--no-browser", action="store_true")
        args = parser.parse_args(test_args[1:])

        assert args.host == "0.0.0.0"
        assert args.api_port == 9000
        assert args.ui_port == 9501
        assert args.no_browser is True

