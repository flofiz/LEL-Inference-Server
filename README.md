# LEL-Inference-Server

Serveur d'inférence HTR (Handwritten Text Recognition) basé sur **Ray Serve** + **vLLM** + **Qwen2.5-VL-3B**.

---

## Architecture

```
Client HTTP
    │
    ▼
Nginx (reverse proxy, TLS)
    │
    ▼
Ray Serve HTTP (port 8000)
    │
    ▼
VLLMPredictDeployment  ←── @serve.deployment (1 GPU)
    │
    ├── MaintenanceMiddleware  (/tmp/maintenance)
    ├── POST /transcribe       → HTREngine.__call__()
    ├── POST /transcribe/stream
    ├── POST /retranscribe
    ├── GET  /health
    └── GET  /model_info
            │
            ▼
        HTREngine
            ├── preprocessing  (crop, resize, cut_image, get_inputs)
            ├── AsyncLLMEngine (vLLM)
            └── postprocessing (postprocess_line → polygones + texte)
```

---

## Démarrage

Le service est géré par **systemd** (`lel.service`). Ray et Ray Serve sont initialisés directement par `main.py` — aucun `ray start --head` externe n'est nécessaire.

```bash
# Démarrer
sudo systemctl start lel

# Arrêter
sudo systemctl stop lel

# Voir les logs
journalctl -f -u lel

# Redémarrer après un changement de code
sudo systemctl restart lel
```

### Maintenance

```bash
# Activer la maintenance (répond 200 + Retry-After: 3600 à toutes les requêtes)
touch /tmp/maintenance

# Désactiver la maintenance
rm /tmp/maintenance
```

---

## Service systemd (`/etc/systemd/system/lel.service`)

```ini
[Service]
User=fizainef
WorkingDirectory=/home/fizainef/LeLRay/src
Environment="PATH=/home/fizainef/LeLRay/.venv/bin:..."
Environment="CUDA_HOME=/usr/local/cuda"
ExecStart=/home/fizainef/LeLRay/.venv/bin/python /home/fizainef/LeLRay/src/main.py
Restart=always
RestartSec=10
```

Le venv est géré avec **uv** (`/home/fizainef/LeLRay/.venv`, Python 3.11).

---

## Modèle

- **Chemin** : `/home/fizainef/LLM/Weights/Qwen2.5-VL_GRPO`
- **Base** : `Qwen2.5-VL-3B-Instruct` fine-tuné SFT puis GRPO (`Paradoxis/Qwen2.5-VL-3B_tsv_grpo-27032026`)
- **Format de sortie** : `|texte|xmin,ymin,xmax,ymax|` (pipe-séparé, sans angle — angle=0 par défaut)
- Pour re-fusionner le modèle :
  ```bash
  cd /home/fizainef/LLM
  python merge_lora.py \
    --base Weights/Qwen2.5-VL_SFT/ \
    --lora Paradoxis/Qwen2.5-VL-3B_tsv_grpo-27032026 \
    --out Weights/Qwen2.5-VL_GRPO
  ```

### Paramètres vLLM (`main.py`)

| Paramètre | Valeur |
|-----------|--------|
| `max_num_seqs` | 90 |
| `max_model_len` | 8192 |
| `max_num_batched_tokens` | 32768 |
| `dtype` | bfloat16 |
| `gpu_memory_utilization` | 0.95 |
| `enable_chunked_prefill` | True |
| `limit_mm_per_prompt` | `{"image": 1}` |

---

## Ray Serve

`main.py` initialise Ray en mode **standalone** (sans cluster externe) :

```python
ray.init(
    _metrics_export_port=8080,   # export métriques Prometheus
    include_dashboard=True,
    dashboard_host="127.0.0.1",  # dashboard Ray accessible sur http://127.0.0.1:8265
)
serve.run(VLLMPredictDeployment.bind(...), blocking=True)
```

- **Dashboard Ray** : `http://127.0.0.1:8265` (local uniquement)
- **HTTP Serve** : port 8000 (derrière Nginx)
- **Métriques** : port 8080 (head node) + ports éphémères pour les workers

Le déploiement `VLLMPredictDeployment` est configuré avec :
- `num_gpus=1` (1 GPU par replica)
- `max_ongoing_requests=100`

---

## Monitoring : Prometheus + Grafana

### Prometheus

- **Binaire** : `/home/fizainef/prometheus-3.11.2.linux-amd64/prometheus`
- **Config** : `/home/fizainef/prometheus-3.11.2.linux-amd64/prometheus.yml`
- **Port** : 9090

Ray écrit automatiquement ses targets dans `/tmp/ray/prom_metrics_service_discovery.json` (file-based service discovery). Prometheus scrape ce fichier toutes les 10 s pour découvrir les endpoints des workers.

```yaml
scrape_configs:
  - job_name: "ray"
    scrape_interval: 10s
    file_sd_configs:
      - files:
          - "/tmp/ray/prom_metrics_service_discovery.json"
```

Targets actifs (exemple) : `172.16.1.187:8080` (head), `172.16.1.187:44217`, `172.16.1.187:44227` (workers).

Métriques disponibles : ~208 métriques `ray_*` dont :
- `ray_cluster_active_nodes` — nœuds actifs
- `ray_serve_deployment_replica_healthy` — santé des replicas
- `ray_serve_num_http_requests_total` — requêtes HTTP
- `ray_serve_deployment_processing_latency_ms_*` — latence
- `ray_component_cpu_percentage`, `ray_component_rss_mb` — ressources

```bash
# Démarrer Prometheus
cd /home/fizainef/prometheus-3.11.2.linux-amd64
./prometheus --config.file=prometheus.yml &

# Vérifier les targets
curl http://localhost:9090/api/v1/targets

# Requête exemple
curl 'http://localhost:9090/api/v1/query?query=ray_serve_deployment_replica_healthy'
```

### Grafana

- **Binaire** : `/home/fizainef/grafana-v11.6.1/`
- **Port** : 3000
- **Credentials** : `admin` / `giN!M3yqG&?HPeDK`

Datasource Prometheus configurée (`uid=dfkcwax8nb400f`, url=`http://localhost:9090`).

**4 dashboards importés** (générés depuis le code Ray) :

| Dashboard | URL | Panels |
|-----------|-----|--------|
| Default Dashboard | `/d/rayDefaultDashboard` | 42 |
| Serve Dashboard | `/d/rayServeDashboard` | 22 |
| Serve Deployment Dashboard | `/d/rayServeDeploymentDashboard` | 15 |
| Data Dashboard | `/d/rayDataDashboard` | 18 |

Pour régénérer les dashboards après une mise à jour de Ray :

```bash
cd /home/fizainef/LeLRay
.venv/bin/python3 - << 'EOF'
from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
    generate_default_grafana_dashboard,
    generate_serve_grafana_dashboard,
    generate_serve_deployment_grafana_dashboard,
    generate_data_grafana_dashboard,
)
import json, requests, os

GRAFANA_URL = "http://localhost:3000"
AUTH = ('admin', 'giN!M3yqG&?HPeDK')
DATASOURCE_UID = "dfkcwax8nb400f"

for title, func in [
    ('Default Dashboard', generate_default_grafana_dashboard),
    ('Serve Dashboard', generate_serve_grafana_dashboard),
    ('Serve Deployment Dashboard', generate_serve_deployment_grafana_dashboard),
    ('Data Dashboard', generate_data_grafana_dashboard),
]:
    content = [r for r in func() if isinstance(r, str) and r.strip().startswith('{')][0]
    dashboard = json.loads(content)
    dashboard_str = json.dumps(dashboard).replace('"${datasource}"', f'{{"type":"prometheus","uid":"{DATASOURCE_UID}"}}')
    dashboard = json.loads(dashboard_str)
    dashboard.pop('id', None)
    dashboard['title'] = title
    r = requests.post(f"{GRAFANA_URL}/api/dashboards/db", json={"dashboard": dashboard, "overwrite": True, "folderId": 0}, auth=AUTH)
    print(f"{'✓' if r.status_code==200 else '✗'} {title}: {r.json().get('url','')}")
EOF
```

---

## Nginx (reverse proxy HTTP/2 + TLS)

Nginx se trouve devant Ray Serve et gère TLS + HTTP/2 côté client. Ray Serve reste en HTTP/1.1 sur `127.0.0.1:8000`.

```
Client (HTTPS / HTTP2)  ──→  nginx :443 (TLS terminé)  ──→  Ray Serve 127.0.0.1:8000 (HTTP/1.1)
```

### Certificats

Les certificats existants sont dans `/home/fizainef/LLM/cert.pem` et `/home/fizainef/LLM/key.pem`.

Pour en regénérer :
```bash
openssl req -x509 -newkey rsa:4096 \
  -keyout /home/fizainef/LLM/key.pem \
  -out /home/fizainef/LLM/cert.pem \
  -days 365 -nodes \
  -subj "/CN=ton-domaine.com"
```

### Config `/etc/nginx/nginx.conf`

La config nginx principale se trouve dans `/etc/nginx/nginx.conf`. Le bloc HTTPS existant doit avoir `http2` dans la directive `listen` (ancienne syntaxe, nginx < 1.25.1) :

```nginx
server {
    listen 443 ssl http2;   # http2 dans listen pour les versions < 1.25.1
    server_name _;

    ssl_certificate     /home/fizainef/LLM/cert.pem;
    ssl_certificate_key /home/fizainef/LLM/key.pem;

    client_max_body_size 100M;

    location / {
        proxy_pass http://ray_backend;   # upstream 127.0.0.1:8000
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        proxy_connect_timeout 300s;
        proxy_read_timeout    300s;
        proxy_send_timeout    300s;

        proxy_buffering off;
        proxy_set_header X-Accel-Buffering no;
    }
}
```

### Appliquer la modification HTTP/2

```bash
# Ajouter http2 à la directive listen (si pas déjà fait)
sudo sed -i 's/listen 443 ssl;/listen 443 ssl http2;/' /etc/nginx/nginx.conf
sudo nginx -t && sudo systemctl reload nginx
```

### Commandes utiles

```bash
# Vérifier la config
sudo nginx -t

# Recharger après modification
sudo systemctl reload nginx

# Vérifier la version (pour choisir la directive http2)
nginx -v
```

> **Note** : `serve.start(http_options={"host": "127.0.0.1", "port": 8000})` dans `main.py` garantit que Ray n'est pas exposé directement sur le réseau.

---

## Développement

```bash
# Activer le venv
source /home/fizainef/LeLRay/.venv/bin/activate

# Installer / synchroniser les dépendances
cd /home/fizainef/LeLRay
uv sync

# Lancer en développement (sans systemd)
cd src
python main.py

# Lancer un test client
python client.py
```

Les logs Ray Serve (erreurs d'inférence, stack traces) sont dans :
```
/tmp/ray/session_latest/logs/serve/
```
Les logs systemd (démarrage, arrêt) sont dans le journal :
```bash
journalctl -u lel --since "1 hour ago"
```

current promete :
To stop Prometheus, use the command: `ray metrics shutdown-prometheus`, 'kill 220892', or if you need to force stop, use 'kill -9 220892'.