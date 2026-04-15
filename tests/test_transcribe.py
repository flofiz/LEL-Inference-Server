"""
Tests d'intégration des endpoints /transcribe et /transcribe/stream.

Ces tests requièrent un serveur HTR en cours d'exécution.
Lancer avec :
    pytest tests/test_transcribe.py -v
    pytest tests/test_transcribe.py -v --base-url https://monserveur:443 --image-path /chemin/image.jpg

Format attendu des réponses
---------------------------
Stream (/transcribe/stream) — lignes NDJSON :
  1. {"metadata": {"imageWidth": int, "imageHeight": int, "model": str}}
  2+. {"label": str, "points": [[int, int], ...], "char_perplexity": float, "line_perplexity": float}
  N. {"metadata": {"metrics": {"file_perplexity": float, "char_perplexity": float}}}

Non-stream (/transcribe) — JSON :
  {
    "imageWidth": int, "imageHeight": int, "model": str,
    "metrics": {"file_perplexity": float, "char_perplexity": float},
    "shapes": [{"label": str, "points": [[int, int], ...],
                "char_perplexity": float, "line_perplexity": float}, ...]
  }
"""
import json
import math
from io import BytesIO
from typing import Any

import pytest
import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multipart_payload(image_bytes: bytes) -> dict:
    return {"image": ("test_image.png", BytesIO(image_bytes), "image/png")}


def _is_finite_float(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _assert_shape(shape: dict) -> None:
    """Valide la structure d'un objet shape."""
    assert isinstance(shape, dict), f"Shape devrait être un dict, reçu : {type(shape)}"
    assert "label" in shape, "Shape manque 'label'"
    assert "points" in shape, "Shape manque 'points'"
    assert "char_perplexity" in shape, "Shape manque 'char_perplexity'"
    assert "line_perplexity" in shape, "Shape manque 'line_perplexity'"

    assert isinstance(shape["label"], str), "'label' devrait être une str"
    assert isinstance(shape["points"], list) and len(shape["points"]) > 0, \
        "'points' devrait être une liste non vide"
    for point in shape["points"]:
        assert len(point) == 2, f"Chaque point devrait avoir 2 coordonnées, reçu : {point}"

    assert _is_finite_float(shape["char_perplexity"]), \
        f"'char_perplexity' devrait être un float fini, reçu : {shape['char_perplexity']}"
    assert _is_finite_float(shape["line_perplexity"]), \
        f"'line_perplexity' devrait être un float fini, reçu : {shape['line_perplexity']}"


def _assert_metrics(metrics: dict) -> None:
    """Valide la structure d'un objet metrics."""
    assert isinstance(metrics, dict), f"Metrics devrait être un dict, reçu : {type(metrics)}"
    assert "file_perplexity" in metrics, "Metrics manque 'file_perplexity'"
    assert "char_perplexity" in metrics, "Metrics manque 'char_perplexity'"
    assert _is_finite_float(metrics["file_perplexity"]), \
        f"'file_perplexity' devrait être un float fini, reçu : {metrics['file_perplexity']}"
    assert _is_finite_float(metrics["char_perplexity"]), \
        f"'char_perplexity' devrait être un float fini, reçu : {metrics['char_perplexity']}"


# ---------------------------------------------------------------------------
# Fixtures locales aux tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def stream_response_lines(base_url, http_session, image_bytes) -> list[dict]:
    """Effectue l'appel stream et retourne la liste des objets JSON parsés.

    scope="session" : une seule requête par run pytest, partagée entre toutes
    les classes de tests — garantit que stream et non-stream comparent bien
    des données issues du même batch d'inférences.
    """
    resp = http_session.post(
        f"{base_url}/transcribe/stream",
        files=_multipart_payload(image_bytes),
        stream=True,
        timeout=300,
    )
    assert resp.status_code == 200, f"Statut inattendu : {resp.status_code}"
    lines = []
    for raw in resp.iter_lines():
        if raw:
            lines.append(json.loads(raw.decode("utf-8")))
    return lines


@pytest.fixture(scope="session")
def non_stream_response(base_url, http_session, image_bytes) -> dict:
    """Effectue l'appel non-stream et retourne le JSON parsé.

    scope="session" : une seule requête par run pytest, partagée entre toutes
    les classes de tests.
    """
    resp = http_session.post(
        f"{base_url}/transcribe",
        files=_multipart_payload(image_bytes),
        timeout=300,
    )
    assert resp.status_code == 200, f"Statut inattendu : {resp.status_code}"
    return resp.json()


# ---------------------------------------------------------------------------
# Tests endpoint stream
# ---------------------------------------------------------------------------

class TestStreamEndpoint:

    def test_returns_at_least_one_line(self, stream_response_lines):
        assert len(stream_response_lines) > 0, "La réponse stream ne contient aucune ligne"

    def test_first_chunk_is_initial_metadata(self, stream_response_lines, image_size):
        first = stream_response_lines[0]
        assert "metadata" in first, f"Premier chunk devrait contenir 'metadata', reçu : {first}"
        meta = first["metadata"]
        width, height = image_size
        assert meta.get("imageWidth") == width, \
            f"imageWidth attendu {width}, reçu {meta.get('imageWidth')}"
        assert meta.get("imageHeight") == height, \
            f"imageHeight attendu {height}, reçu {meta.get('imageHeight')}"
        assert isinstance(meta.get("model"), str) and meta["model"], \
            "'model' devrait être une str non vide"

    def test_last_chunk_contains_metrics(self, stream_response_lines):
        last = stream_response_lines[-1]
        assert "metadata" in last, f"Dernier chunk devrait contenir 'metadata', reçu : {last}"
        assert "metrics" in last["metadata"], \
            f"Dernier chunk metadata devrait contenir 'metrics', reçu : {last['metadata']}"
        _assert_metrics(last["metadata"]["metrics"])

    def test_shape_chunks_have_valid_structure(self, stream_response_lines):
        shape_chunks = [
            c for c in stream_response_lines
            if "metadata" not in c
        ]
        assert len(shape_chunks) > 0, "Aucun chunk de type shape reçu"
        for shape in shape_chunks:
            _assert_shape(shape)

    def test_perplexities_are_non_negative(self, stream_response_lines):
        shape_chunks = [c for c in stream_response_lines if "metadata" not in c]
        for shape in shape_chunks:
            assert shape["char_perplexity"] >= 0, \
                f"char_perplexity négatif : {shape['char_perplexity']}"
            assert shape["line_perplexity"] >= 0, \
                f"line_perplexity négatif : {shape['line_perplexity']}"

    def test_metrics_perplexities_are_positive(self, stream_response_lines):
        metrics = stream_response_lines[-1]["metadata"]["metrics"]
        assert metrics["file_perplexity"] > 0, \
            f"file_perplexity devrait être > 0, reçu : {metrics['file_perplexity']}"
        assert metrics["char_perplexity"] > 0, \
            f"char_perplexity devrait être > 0, reçu : {metrics['char_perplexity']}"


# ---------------------------------------------------------------------------
# Tests endpoint non-stream
# ---------------------------------------------------------------------------

class TestNonStreamEndpoint:

    def test_response_has_required_top_level_keys(self, non_stream_response):
        for key in ("imageWidth", "imageHeight", "model", "metrics", "shapes"):
            assert key in non_stream_response, f"Clé manquante dans la réponse : '{key}'"

    def test_image_dimensions_match(self, non_stream_response, image_size):
        width, height = image_size
        assert non_stream_response["imageWidth"] == width, \
            f"imageWidth attendu {width}, reçu {non_stream_response['imageWidth']}"
        assert non_stream_response["imageHeight"] == height, \
            f"imageHeight attendu {height}, reçu {non_stream_response['imageHeight']}"

    def test_model_is_non_empty_string(self, non_stream_response):
        assert isinstance(non_stream_response["model"], str) and non_stream_response["model"], \
            "'model' devrait être une str non vide"

    def test_metrics_structure(self, non_stream_response):
        _assert_metrics(non_stream_response["metrics"])

    def test_shapes_is_non_empty_list(self, non_stream_response):
        shapes = non_stream_response["shapes"]
        assert isinstance(shapes, list), "'shapes' devrait être une liste"
        assert len(shapes) > 0, "'shapes' ne devrait pas être vide"

    def test_shapes_have_valid_structure(self, non_stream_response):
        for shape in non_stream_response["shapes"]:
            _assert_shape(shape)

    def test_shapes_perplexities_are_non_negative(self, non_stream_response):
        for shape in non_stream_response["shapes"]:
            assert shape["char_perplexity"] >= 0, \
                f"char_perplexity négatif : {shape['char_perplexity']}"
            assert shape["line_perplexity"] >= 0, \
                f"line_perplexity négatif : {shape['line_perplexity']}"

    def test_metrics_perplexities_are_positive(self, non_stream_response):
        metrics = non_stream_response["metrics"]
        assert metrics["file_perplexity"] > 0, \
            f"file_perplexity devrait être > 0, reçu : {metrics['file_perplexity']}"
        assert metrics["char_perplexity"] > 0, \
            f"char_perplexity devrait être > 0, reçu : {metrics['char_perplexity']}"


# ---------------------------------------------------------------------------
# Tests de cohérence stream <-> non-stream
# ---------------------------------------------------------------------------

class TestStreamNonStreamConsistency:
    """
    Vérifie que les deux endpoints produisent des résultats cohérents
    pour la même image. Les valeurs de perplexité doivent être identiques
    car les deux chemins partagent la même logique via stream_results.

    Note sur l'ordre des shapes
    ---------------------------
    Pour les documents multi-pages, stream_multiple_results interleave les
    shapes de chaque page via une queue asyncio (ordre non garanti), tandis
    que le chemin non-stream concatène les pages dans l'ordre strict.
    Les comparaisons sont donc faites sur des listes triées par label.

    Note sur le non-déterminisme GPU
    ----------------------------------
    Même avec temperature=0, les CUDA kernels peuvent varier légèrement d'une
    inférence à l'autre (ordre des opérations flottantes, NCCL, cuDNN). Les
    deux requêtes sont donc des inférences distinctes : une tolérance de 5 %
    est appliquée sur les perplexités par shape, et 2 % sur les métriques
    globales (moyennées sur plus de tokens, donc plus stables).
    """

    @staticmethod
    def _sorted_shapes(shapes: list[dict]) -> list[dict]:
        return sorted(shapes, key=lambda s: s.get("label", ""))

    def test_same_metadata(self, stream_response_lines, non_stream_response):
        """Les métadonnées image (imageWidth, imageHeight, model) doivent être
        identiques entre les deux endpoints : elles sont dérivées de la même
        image d'entrée et du même moteur, indépendamment du mode de rendu."""
        stream_initial_meta = stream_response_lines[0]["metadata"]
        for key in ("imageWidth", "imageHeight", "model"):
            assert key in non_stream_response, \
                f"Clé '{key}' absente de la réponse non-stream"
            assert stream_initial_meta[key] == non_stream_response[key], (
                f"Métadonnée '{key}' différente — "
                f"stream : {stream_initial_meta[key]}, "
                f"non-stream : {non_stream_response[key]}"
            )

    def test_same_number_of_shapes(self, stream_response_lines, non_stream_response):
        stream_shapes = [c for c in stream_response_lines if "metadata" not in c]
        non_stream_shapes = non_stream_response["shapes"]
        assert len(stream_shapes) == len(non_stream_shapes), (
            f"Nombre de shapes différent — stream : {len(stream_shapes)}, "
            f"non-stream : {len(non_stream_shapes)}"
        )

    def test_same_labels(self, stream_response_lines, non_stream_response):
        stream_labels = sorted(
            c["label"] for c in stream_response_lines if "metadata" not in c
        )
        non_stream_labels = sorted(s["label"] for s in non_stream_response["shapes"])
        assert stream_labels == non_stream_labels, (
            "Les labels diffèrent entre stream et non-stream (après tri).\n"
            f"Stream    : {stream_labels}\n"
            f"Non-stream: {non_stream_labels}"
        )

    def test_shape_perplexities_match(self, stream_response_lines, non_stream_response):
        stream_shapes = self._sorted_shapes(
            [c for c in stream_response_lines if "metadata" not in c]
        )
        non_stream_shapes = self._sorted_shapes(non_stream_response["shapes"])
        for i, (s, ns) in enumerate(zip(stream_shapes, non_stream_shapes)):
            # rel=0.05 : 5 % de tolérance pour absorber le non-déterminisme GPU
            # entre deux inférences séparées (logprobs flottants, CUDA kernels).
            assert s["char_perplexity"] == pytest.approx(ns["char_perplexity"], rel=0.05), (
                f"Shape[{i}] (label='{s['label']}') char_perplexity : "
                f"stream={s['char_perplexity']}, non-stream={ns['char_perplexity']}"
            )
            assert s["line_perplexity"] == pytest.approx(ns["line_perplexity"], rel=0.05), (
                f"Shape[{i}] (label='{s['label']}') line_perplexity : "
                f"stream={s['line_perplexity']}, non-stream={ns['line_perplexity']}"
            )

    def test_global_metrics_match(self, stream_response_lines, non_stream_response):
        stream_metrics = stream_response_lines[-1]["metadata"]["metrics"]
        non_stream_metrics = non_stream_response["metrics"]
        assert stream_metrics["file_perplexity"] == pytest.approx(
            non_stream_metrics["file_perplexity"], rel=0.01
        ), (
            f"file_perplexity : stream={stream_metrics['file_perplexity']}, "
            f"non-stream={non_stream_metrics['file_perplexity']}"
        )
        assert stream_metrics["char_perplexity"] == pytest.approx(
            non_stream_metrics["char_perplexity"], rel=0.01
        ), (
            f"char_perplexity : stream={stream_metrics['char_perplexity']}, "
            f"non-stream={non_stream_metrics['char_perplexity']}"
        )
