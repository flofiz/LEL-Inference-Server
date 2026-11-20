# LEL-Inference-Server

Serveur d'inférence HTR (Handwritten Text Recognition) basé sur **Ray Serve**,
**vLLM 0.34.0** et **Qwen3.5-4B RSWA**.

**Documentation complète : [guide de maintenance RSWA, HTML et démo](docs/guide-maintenance-rswa.md)** —
comparaison avec le serveur d'origine, modifications dans vLLM, paramètres, API, tests et procédure de retour arrière.

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
    ├── POST /transcribe/html
    ├── POST /transcribe/html/stream
    ├── GET  /demo
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

### Vérifier la migration vLLM 0.34 sur l'A40

Depuis la racine du dépôt, après `uv sync`, avec un GPU disponible :

```bash
uv run --with pytest pytest tests/test_rswa_kernel.py -v -k 'not recycled_gap'
```

Ces tests ne démarrent pas Ray et ne chargent aucun modèle. Ils vérifient les
imports du serveur et de l'extension vLLM, une opération CUDA BF16 et le masque
RSWA natif de Triton contre une référence FP32. Sous Windows, le test CPU
des frontières du masque peut s'exécuter sans installer le projet :

```powershell
uv run --no-project --with pytest --with requests --with pillow pytest tests/test_rswa_kernel.py -v
```

Pour tester le patch RSWA sur l'A40, dans l'environnement de test et serveur
arrêté, appliquer puis vérifier le patch Python de la roue vLLM (sans
recompilation). Le script refuse toute version ou source différente de celle
auditée et peut être relancé sans réappliquer le patch.

```bash
uv run --no-sync python src/inference/rswa_patch.py --apply --mtp3
uv run --no-sync python src/inference/rswa_patch.py --mtp3
uv run --no-sync --with pytest pytest tests/test_rswa_kernel.py -v
```

`recycled_gap` doit maintenant passer avec des NaN dans un bloc invisible.
Le patch ne modifie pas les checkpoints. Ray Serve active RSWA uniquement si
le manifeste `rswa_training.json` et les métadonnées dans `config.json` sont
cohérents. Le mode RSWA exige un Qwen3.5 multimodal BF16 non quantifié et un
seul GPU. Ray Serve utilise MTP `K=3` par défaut.

Pour valider le vrai checkpoint avant le démarrage du serveur, sur l'A40 libre :

```bash
HTR_MTP_TOKENS=3 HTR_RSWA_TEST_MODEL=/chemin/checkpoint-rswa uv run --no-sync --with pytest \
  pytest tests/test_rswa_checkpoint.py -v -s --image-path /chemin/page.jpg
```

Le test force une sortie assez longue pour laisser au moins un bloc évictable
et vérifie que les log-probabilités générées sont finies. Rechercher `RSWA target
ran`, `RSWA draft ran` et `RSWA first KV eviction` dans les logs.
La sortie brute n'est pas enregistrée par défaut. Pour la conserver, ajouter
`HTR_RSWA_TEST_OUTPUT=./rswa-generation.txt` devant la commande. Ce texte peut
se répéter après la fin naturelle de la transcription : le test ignore EOS.

Pour démarrer Ray Serve avec le checkpoint RSWA par défaut :

```bash
.venv/bin/python src/main.py
```

Utiliser directement le Python du venv (comme dans systemd). Lancé via
`uv run`, Ray peut réexécuter `uv run` pour ses workers depuis un autre
répertoire et créer un venv sans les dépendances du serveur.

Les réglages de démarrage sont dans [`server.toml`](server.toml). Pour modifier
uniquement le chemin sur une machine, créer `server.local.toml` à la racine
(ignoré par Git) :

```toml
[engine]
model = "/chemin/absolu/du/checkpoint"
```

Les clés locales remplacent les clés de défaut. `HTR_CONFIG_PATH` peut pointer
vers un autre fichier de surcharge, par exemple sous `/etc`, et les variables
existantes (`HTR_MODEL_PATH`, `HTR_MTP_TOKENS`, etc.) restent prioritaires.
Redémarrer le service après changement. RSWA est obligatoire par
défaut : un manifeste incomplet ou absent, un patch absent ou un cache
incompatible fait échouer l'initialisation du réplica, sans repli en full
attention. Pour servir volontairement un checkpoint standard, définir aussi
`HTR_REQUIRE_RSWA=0`.

### Préparer et vérifier MTP3 avec RSWA

Serveur arrêté, dans le venv de l'A40 :

```bash
.venv/bin/python src/inference/rswa_patch.py --apply --mtp3
HTR_MTP_TOKENS=3 HTR_RSWA_TEST_MODEL=/chemin/checkpoint-rswa \
  uv run --no-sync --with pytest pytest tests/test_rswa_checkpoint.py -v -s \
  --image-path ../FRAD021_B_012126_00013.jpg
```

L'installation conserve la variante `pruned` si elle est déjà en place.
MTP3 utilise la même couche MTP en trois étapes séquentielles ; le patch
transmet le préfixe multimodal complet lors de chaque reconstruction des
métadonnées. Après le test, le service démarre en MTP3 sans variable
supplémentaire. Le réglage s'applique au moteur entier, donc aussi aux routes
TSV et HTML non streamées. Définir explicitement `HTR_MTP_TOKENS=1` dans
l'environnement du service permet de revenir à MTP1 sans réinstaller vLLM.

### Thinking Qwen3.5

Le serveur passe `enable_thinking=False` au chat template Qwen3.5 par défaut,
avec ou sans RSWA. Pour le réactiver, définir `HTR_ENABLE_THINKING=1` au
démarrage. Le template Qwen officiel insère un bloc `<think>` déjà fermé dans
le prompt lorsque le thinking est désactivé. Si un checkpoint utilise un
template personnalisé, celui-ci doit gérer l'option `enable_thinking`.

### Benchmark RSWA avant/après optimisation Triton

Sur l'A40, arrêter Ray Serve et lancer depuis la racine du dépôt, dans le
venv où le patch RSWA est déjà installé :

```bash
.venv/bin/python scripts/benchmark_rswa.py \
  --model ../Weights/Qwen3.5-4B-06112026-sft-html-convert-rswa-w512_grpo \
  --normal-image ../FRAD021_B_012126_00013.jpg \
  --long-image ../FRAD021_10M106_05_0004.jpg \
  --long-lengths 8192 16384 --concurrency 1 4 \
  --output rswa-baseline.json
```

La double page est découpée selon les mêmes règles que le serveur. Chaque
demi-page est mesurée avec les longueurs demandées **par requête**, EOS ignoré.
La page normale garde son comportement de génération habituel, sous le plafond
de `--normal-max-tokens`. Le contexte du benchmark vaut 24 576 tokens par
défaut, pour laisser de la place au prompt multimodal en plus des 16 384
tokens de sortie. Ceci ne modifie ni les limites du serveur ni le format HTR.

Le rapport JSON contient les temps au premier token et de génération, les
débits par requête et agrégés, et le pic d'occupation KV observé. Il est mis à
jour après chaque cas. Pour un premier essai plus court, utiliser
`--long-lengths 8192 --concurrency 1`. Pour la comparaison avant/après, garder
les mêmes options et images, et écrire les deux rapports dans des fichiers
distincts. Les temps sont mesurés au niveau du moteur, hors HTTP et préparation
initiale des images.

Une fois le rapport de référence sauvegardé, toujours serveur arrêté :

```bash
.venv/bin/python src/inference/rswa_patch.py --apply --variant pruned
uv run --no-sync --with pytest pytest tests/test_rswa_kernel.py -v
```

Rejouer ensuite le benchmark avec **les mêmes paramètres**, en ajoutant
`--label pruned` et en utilisant `--output rswa-pruned.json`. La variante
`pruned` ne modifie que le parcours des tuiles Triton, pas le masque ni
l'éviction KV. Pour revenir au noyau de référence sans réinstaller vLLM :

```bash
.venv/bin/python src/inference/rswa_patch.py --apply --variant baseline
```

Pour vérifier le serveur et une vraie transcription après redémarrage :

```bash
uv run --with pytest pytest tests/test_transcribe.py -v --image-path /chemin/page.jpg
```

### Transcription HTML

`POST /transcribe/html` accepte le même upload multipart `image` que
`/transcribe`, qui conserve son format TSV. Le nouveau chemin utilise le prompt
HTML du checkpoint et retourne du JSON, sans convertir le HTML en polygones :

```bash
curl --fail-with-body -sS -F "image=@../FRAD021_10M106_05_0004.jpg" \
  http://127.0.0.1:8000/transcribe/html -o transcription-html.json
```

La réponse contient `imageWidth`, `imageHeight`, `model` et `pages`. Chaque
entrée de `pages` contient `html`, `offsetX`, `width`, `height` et `finishReason`.
Les fragments sont dans l'ordre gauche → droite si l'image est découpée. Les
coordonnées `coord` restent normalisées sur `[0,1000]` dans le repère de **leur
fragment** ; `offsetX` (en pixels) donne sa position dans l'image d'origine.
Seules les clôtures Markdown externes sont retirées. Le HTML reste une sortie
du modèle : le filtrer avant de l'insérer dans un navigateur.

La génération s'arrête normalement sur EOS, sinon au budget de **16 384 tokens
par fragment**, dans un contexte de **24 576 tokens**. `finishReason: "length"`
indique que la sortie peut être incomplète.

`POST /transcribe/html/stream` accepte le même upload et renvoie du NDJSON.
L'événement `metadata` décrit les fragments et leur géométrie. Chaque `delta`
transmet un morceau de HTML brut (y compris les clôtures Markdown éventuelles)
avec son `pageIndex` **à partir de zéro** ; `done` ou `error` termine le
fragment, et `complete` termine le flux. Les fragments peuvent progresser
en parallèle, mais les morceaux de chaque fragment restent ordonnés.

```bash
curl -N -sS -F "image=@../FRAD021_10M106_05_0004.jpg" \
  https://lel.cotedor.fr/transcribe/html/stream
```

L'API demande au reverse proxy de ne pas mettre cette réponse en tampon
(`X-Accel-Buffering: no`). Si la démo n'affiche les données qu'à la fin,
comparer le port local et l'adresse publique avec `curl -N`.

### Client local de démonstration HTML

Depuis la racine du dépôt, sur la machine qui présente la démo :

```bash
python scripts/demo_transcription_html.py
```

Le navigateur s'ouvre sur `http://127.0.0.1:8765/`. Déposer un JPEG, PNG ou WebP,
ou cliquer sur **Importer depuis mes fichiers** : le choix de l'image lance
immédiatement la transcription.
Le client utilise le serveur public `https://lel.cotedor.fr` via un relais
local ; aucun changement du serveur, de CORS ou du venv d'inférence n'est
nécessaire. Python seul suffit. Le client utilise la route HTML streamée :
les segments et la transcription apparaissent au fil des morceaux reçus.
Le navigateur ferme provisoirement les balises ouvertes (par exemple un
tableau, à compléter au prochain morceau). À la fin, l'aperçu peut toujours
être téléchargé pour une présentation hors ligne.
L'import accepte plusieurs images à la fois, avec 128 requêtes simultanées
au maximum côté client. Une double page peut déclencher deux générations ;
le GPU peut en mettre une partie en attente. La barre de miniatures à gauche sert à changer
d'image, à en ajouter (`+`), et à fermer une image (`×`), les images cochées
ou toutes les images. **PDF A4** ouvre la boîte d'impression du navigateur
pour la transcription de l'onglet actif, sans le scan ni les overlays.

Pour partager la démo sans Python, privilégier
[`https://lel.cotedor.fr/demo`](https://lel.cotedor.fr/demo) : la page et l'API
ont la même origine et n'ont pas besoin de CORS. On peut aussi envoyer
[`demo-lettres-en-lumieres.html`](demo-lettres-en-lumieres.html). S'il est ouvert
depuis le disque et que le proxy bloque l'accès à l'API, cliquer sur **Ouvrir
la démo en ligne**. Après une modification de l'interface,
`python scripts/build_portable_demo.py` reconstruit le fichier à partager et
servi par `/demo`.

### Benchmark HTTP par paliers

Sur l'A40, avec le service lancé et sans autre charge de production :

```bash
.venv/bin/python scripts/benchmark_html_http.py \
  --image ../FRAD021_10M106_05_0004.jpg --output rswa-http-benchmark.json \
  --levels 1 4 8 16 32 48 64 96 128
```

Le script appelle `127.0.0.1:8000` par défaut, sans navigateur ni proxy. Il
mesure le temps jusqu'au premier morceau HTML, la durée totale et les nombres
d'images et de fragments terminés. Il arrête l'escalade et annule les requêtes
restantes au premier échec ou dépassement de délai (60 s jusqu'au premier
delta, 600 s par image, modifiables). Utiliser `journalctl -fu lel` et
`nvidia-smi -l 1` côté A40 pour expliquer une saturation. `--public` ajoute
le passage par le proxy HTTPS ; ne l'utiliser qu'en créneau de test convenu.
`--http1` force HTTP/1.1 pour reproduire les mesures publiques. `--resize`
ramène l'image au budget Qwen3.5 (2 048 000 pixels par moitié, chevauchement
inclus). Le rapport distingue le temps d'envoi vu par le client du délai
après l'envoi ; ce dernier inclut encore le proxy et le prétraitement. Pour un
lot varié, passer plusieurs fichiers à `--image` : ils sont utilisés en boucle
à l'intérieur de chaque palier.

### Matrice locale A40 : K=0/1/2/3, concurrence et budget par itération

Cette mesure utilise le moteur **sans Ray HTTP, sans upload et sans proxy**.
Arrêter `lel`, vérifier que l'A40 est libre et que le patch multi-étapes est
installé (il est partagé par K=2 et K=3) :

```bash
sudo systemctl stop lel
.venv/bin/python src/inference/rswa_patch.py --mtp3

.venv/bin/python scripts/benchmark_rswa_matrix.py \
  --model /home/fizainef/Weights/Qwen3.5-4B-c400-wsp-hybrid-60-50 \
  --normal-image ../FRAD021_B_012126_00013.jpg \
  --long-image ../FRAD021_10M106_05_0004.jpg \
  --out-dir benchmark-a40

sudo systemctl start lel
```

`--normal-image` accepte plusieurs pages. Par défaut la matrice compare les
quatre K et soumet **1, 2, 4, 8, 16, 32, 48, 64, 80, 90, 96 et 128 fragments
vLLM** par vague (pas des images doubles). Les demi-pages de l'image longue
sont alternées. `max_num_seqs` reste fixé à 90 par défaut pour comparer avec
le serveur ; les fragments au-delà peuvent attendre. `--batched-tokens 4096
8192 16384` compare aussi les budgets de tokens par itération ; le défaut est
`16384`. Ne pas confondre ce budget et le nombre de fragments en vol. Le
checkpoint, RSWA et la variante Triton restent identiques ; K=0 désactive
seulement la spéculation. Les réglages de production ne sont pas modifiés.

La matrice complète peut prendre longtemps. Pour un premier passage, ajouter
`--suite normal --levels 1 8 32 64 90` ; ensuite comparer les modes proches
avec `--repeats 3`. Le profil long s'arrête naturellement par défaut ;
`--long-lengths 8192 16384` force des longueurs identiques pour isoler le débit
en décodage. Chaque mode redémarre son propre moteur, écrit `engine.log` et
un `report.json` mis à jour par vague ; la matrice produit `summary.csv` et
`matrix.json`. `--gpu` choisit le GPU physique (`0` par défaut) pour CUDA et
`nvidia-smi`. `--wave-timeout` borne chaque vague à 30 minutes par défaut.

**Interprétation :**

- **Capacité KV** : cache proche de 100 % et préemptions qui augmentent. La
  VRAM réservée dans `nvidia-smi` ne suffit pas à le prouver : vLLM préalloue
  son cache.
- **Limite de planification** : séquences actives collées à `max_num_seqs`,
  file d'attente en hausse et cache loin d'être plein.
- **Limite côté GPU** : débit qui plafonne sans pression KV. L'activité GPU
  et le contrôleur mémoire, échantillonnés via `nvidia-smi`, orientent le
  diagnostic mais ne mesurent ni le taux d'utilisation des Tensor Cores ni
  la bande passante effective. Pour trancher calcul contre bande passante,
  profiler une courte vague représentative avec Nsight et des compteurs DRAM
  et SM ; ne pas utiliser ces exécutions profilées comme scores de référence.

Comparer `fragments_min`, `tokens_s`, `ttft_p95_s` et `acceptance_fraction`
pour chaque situation, et écarter les essais naturels tronqués ou incomplets
si l'objectif est la vitesse de transcription complète.

---

## Modèle

- **Checkpoint par défaut** : `/home/fizainef/Weights/Qwen3.5-4B-c400-wsp-hybrid-60-50`
- **Sorties disponibles** : TSV converti en polygones ou HTML par fragment.

### Ancien checkpoint Qwen2.5 (historique)

- **Chemin historique** : `/home/fizainef/LLM/Weights/Qwen2.5-VL_GRPO`
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

### Paramètres vLLM (`server.toml`)

| Paramètre | Valeur |
|-----------|--------|
| `max_num_seqs` | 90 |
| `max_model_len` | 24576 |
| `max_num_batched_tokens` | 16384 |
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
- `max_ongoing_requests=128` (plafond Ray par réplica, distinct des `max_num_seqs=90` de vLLM)

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
