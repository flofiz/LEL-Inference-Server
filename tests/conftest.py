"""
Fixtures partagées pour les tests d'intégration du serveur HTR.

Variables d'environnement configurables :
  HTR_BASE_URL   URL de base du serveur  (défaut : https://localhost:443)
  HTR_IMAGE_PATH Chemin vers l'image de test (défaut : /home/fizainef/FRAD034_C07228_00033.jpg)
"""
import os
from io import BytesIO

import pytest
import requests
import urllib3
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_BASE_URL = "https://localhost:443"
DEFAULT_IMAGE_PATH = "/home/fizainef/FRAD034_C07228_00033.jpg"


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default=os.environ.get("HTR_BASE_URL", DEFAULT_BASE_URL),
        help="URL de base du serveur HTR",
    )
    parser.addoption(
        "--image-path",
        default=os.environ.get("HTR_IMAGE_PATH", DEFAULT_IMAGE_PATH),
        help="Chemin de l'image à envoyer pour les tests",
    )


@pytest.fixture(scope="session")
def base_url(request) -> str:
    return request.config.getoption("--base-url").rstrip("/")


@pytest.fixture(scope="session")
def http_session() -> requests.Session:
    """Session HTTP avec SSL désactivé (certificat auto-signé)."""
    session = requests.Session()
    session.verify = False
    return session


@pytest.fixture(scope="session")
def image_bytes(request) -> bytes:
    """Charge l'image de test en bytes PNG."""
    path = request.config.getoption("--image-path")
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def image_size(request) -> tuple[int, int]:
    """Retourne (width, height) de l'image de test."""
    path = request.config.getoption("--image-path")
    return Image.open(path).size
