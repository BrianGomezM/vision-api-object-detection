import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile

from app.routes import evaluation
from app.utils.uploads import read_upload_limited


@pytest.mark.parametrize("path", [
    "../.env",
    "../app/main.py",
    "test_images/../../.env",
    "/etc/passwd",
    "C:/Windows/win.ini",
    "http://169.254.169.254/latest/meta-data/",
    "05_sala.txt",
    "../yolo26s.pt",
])
def test_resolve_test_image_rechaza_rutas_fuera_de_test_images(path):
    with pytest.raises(HTTPException) as e:
        evaluation._resolve_test_image(path)
    assert e.value.status_code == 400


@pytest.mark.parametrize("path", ["05_sala.jpg", "test_images/05_sala.jpg", "sub/escena.PNG"])
def test_resolve_test_image_acepta_imagenes_dentro(path):
    p = evaluation._resolve_test_image(path)
    assert evaluation._TEST_IMAGES_DIR in p.parents


def test_autotest_apunta_solo_a_loopback():
    assert evaluation._SELF_TEST_BASE_URL.startswith(("http://127.0.0.1", "http://localhost"))


def _upload(n: int) -> UploadFile:
    return UploadFile(file=io.BytesIO(b"\0" * n), filename="a.jpg")


def test_upload_dentro_del_limite():
    assert len(asyncio.run(read_upload_limited(_upload(100), max_bytes=100))) == 100


def test_upload_supera_el_limite_413():
    with pytest.raises(HTTPException) as e:
        asyncio.run(read_upload_limited(_upload(101), max_bytes=100))
    assert e.value.status_code == 413
