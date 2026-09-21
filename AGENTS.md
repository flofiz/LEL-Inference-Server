# LeLRay — Agent Instructions

Serveur d'inférence HTR (Handwritten Text Recognition) basé sur Ray Serve + vLLM + Qwen2.5-VL-3B.

## Architecture

```
src/
├── main.py                  # Entrypoint : ray.init() + serve.run()
├── inference/engine.py      # HTREngine : wraps vLLM AsyncLLMEngine
├── processing/
│   ├── preprocessing.py     # crop_and_get_boxes, reshape_image, get_inputs
│   └── postprocessing.py    # postprocess_line → polygones + texte
├── templates/               # Prompts système et tâches HTR (SYSTEM, PROMPT)
└── api/routes.py            # Routes FastAPI (non utilisées directement, voir main.py)
```

**Stack** : Python 3.11, uv venv (`.venv/`), Ray 2.55.1, vLLM 0.19.1, torch 2.10+cu130.

## Build & Test

```bash
# Installer les dépendances
cd /home/fizainef/LeLRay
uv sync

# Lancer le serveur en développement (sans systemd)
cd src && python main.py

# Tests d'intégration (serveur doit être démarré)
cd /home/fizainef/LeLRay
uv run pytest tests/ -v
# Ou avec paramètres custom :
HTR_BASE_URL=https://localhost:443 uv run pytest tests/ -v
```

## Conventions

### Format de sortie du modèle
Le modèle génère des lignes au format pipe : `|texte|xmin,ymin,xmax,ymax|`
`postprocess_line` dans `processing/postprocessing.py` parse ce format. L'angle est 0 par défaut (pas dans la sortie du modèle).

### Logging
- Logs Ray Serve / vLLM (erreurs d'inférence, stack traces) → `/tmp/ray/session_latest/logs/serve/`
- Logs systemd (démarrage/arrêt service) → `journalctl -u lel`
- Les logs vLLM sont redirigés vers `ray.serve` via `_VLLMToRayHandler` dans `engine.py`

### Ray : initialisation standalone
`ray.init()` est appelé dans `main.py` — ne jamais lancer `ray start --head` en parallèle.
En cas de cluster gelé : `ray stop --force` puis redémarrer le service.

Ray Serve écoute uniquement sur `127.0.0.1:8000` (loopback). Nginx gère TLS + HTTP/2 en frontal sur le port 443 et proxy vers cette adresse. Ne jamais binder Ray sur `0.0.0.0`.

### get_tokenizer()
Dans vLLM 0.19.1, `engine.get_tokenizer()` est **synchrone** — ne pas utiliser `await`.

### Maintenance
Créer `/tmp/maintenance` pour bloquer toutes les requêtes (répond 200 + Retry-After).

## Fichiers importants hors du projet

| Fichier | Rôle |
|---------|------|
| `/etc/systemd/system/lel.service` | Service systemd (uv venv, CUDA_HOME, Restart=always) |
| `/home/fizainef/LLM/Weights/Qwen2.5-VL_GRPO/` | Modèle fusionné (SFT + GRPO LoRA) |
| `/home/fizainef/LLM/Weights/Qwen2.5-VL_GRPO/config.json` | Doit contenir `"tie_word_embeddings": true` |
| `/home/fizainef/prometheus-3.11.2.linux-amd64/` | Prometheus, port 9090 |
| `/home/fizainef/grafana-v11.6.1/` | Grafana, port 3000 |
| `/tmp/ray/prom_metrics_service_discovery.json` | Targets Ray pour Prometheus (auto-généré) |
| `/etc/nginx/nginx.conf` | Config nginx HTTP/2 + TLS (`listen 443 ssl http2`) — reverse proxy vers 127.0.0.1:8000 |
| `/home/fizainef/LLM/cert.pem` | Certificat TLS nginx |
| `/home/fizainef/LLM/key.pem` | Clé privée TLS nginx |

## Monitoring

Prometheus scrape les métriques Ray via file-based service discovery (toutes les 10 s).
Grafana (`http://<serveur>:3000`) expose 4 dashboards : Default, Serve, Serve Deployment, Data.
Voir `README.md` pour la commande de régénération des dashboards après mise à jour de Ray.
