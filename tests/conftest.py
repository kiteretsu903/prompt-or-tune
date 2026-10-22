import socket

import pytest


@pytest.fixture(autouse=True)
def block_external_connections(monkeypatch):
    original = socket.socket.connect

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Tests must not make network requests")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
