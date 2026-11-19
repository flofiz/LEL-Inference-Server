# Guide de maintenance — RSWA, HTML et démo

Ce guide compare le **dépôt** avant l'intervention (`e9c81e5`) et après la livraison de la démo publique (`f5447e9`, branche `feat/rswa`). Il ne prétend pas reconstruire d'éventuelles modifications locales non commitées sur le serveur d'origine.

**Configuration actuelle** : depuis cette livraison, les valeurs de démarrage
ont été déplacées dans [`server.toml`](../server.toml). Le modèle par défaut
est maintenant `Qwen3.5-4B-c400-wsp-hybrid-60-50`. Pour une modification
propre à un hôte, utiliser `server.local.toml` (ignoré par Git) ou
`HTR_CONFIG_PATH` ; les variables `HTR_*` historiques restent prioritaires.
Les chemins de poids et mesures ci-dessous décrivent le checkpoint de référence
de la livraison `f5447e9`, pas un benchmark du nouveau checkpoint.

## 1. Voir précisément ce qui a changé

```bash
git log --reverse --oneline e9c81e5..f5447e9
git diff --stat e9c81e5..f5447e9
git diff e9c81e5..f5447e9 -- pyproject.toml src/main.py src/inference/engine.py
git show 2374edb -- src/inference/rswa_patch.py
```

Le diff de livraison couvre 27 fichiers. Trois états sont à distinguer :

1. **Dépôt Git** : configuration, scripts, API, prompts, tests et page de démo. Un `git pull` ne modifie que cet état.
2. **Venv Linux** : `uv sync` installe les paquets ; `rswa_patch.py` modifie ensuite quelques sources Python dans `.venv/lib/python3.11/site-packages/vllm/`. Un changement de roue ou une réinstallation peut effacer ces patchs. Git ne les enregistre pas.
3. **Machine cible** : checkpoint, pilote CUDA, unité systemd, configuration du reverse proxy, images et rapports. Ces éléments ne sont pas déployés par ce dépôt.

| Élément | Avant (`e9c81e5`) | Après (`f5447e9`) |
|---|---|---|
| Moteur | vLLM `0.29`, PyTorch `2.13.0+cu130` | vLLM `0.34.0`, PyTorch `2.14.0+cu130` et roue vLLM patchée en Python |
| Checkpoint par défaut | `/home/fizainef/Weights/Qwen3.5-4B-15092026` | `/home/fizainef/Weights/Qwen3.5-4B-06112026-sft-html-convert-rswa-w512_grpo` |
| Spéculation | MTP `K=2` si le nom du modèle contenait `Qwen3.5` | RSWA obligatoire par défaut, MTP séquentiel `K=3` ; possibilité de revenir à `K=1` |
| Contexte total | 8 192 tokens | 24 576 tokens pour laisser la place au prompt et aux 16 384 tokens de sortie HTML |
| Thinking | Aucun `enable_thinking` explicite | Désactivé par défaut pour Qwen3.5, réactivable |
| TSV | Deux routes, scores dépendants du regroupement des tokens | Mêmes routes et formes de réponse ; calcul corrigé des scores |
| HTML / présentation | Pas de route HTML ni de client de démo | HTML complet, HTML en streaming, aperçu local et démo HTTPS |

Inchangés dans le point d'entrée : un GPU par réplica, `max_num_seqs=90`, `max_num_batched_tokens=16384`, `gpu_memory_utilization=0.95`, BF16, chunked prefill, une image par requête vLLM et écoute Ray Serve sur `127.0.0.1:8000`. `max_ongoing_requests=100` est une limite Ray, pas une garantie de 100 générations actives sur le GPU. Nginx/TLS, Prometheus et Grafana n'ont pas été modifiés. La route historique `/retranscribe` existait déjà mais sa méthode moteur n'a pas été implémentée par ce travail.

## 2. Installation et reproduction sur Linux / A40

Prérequis : dépôt sur `feat/rswa`, `uv`, Python `3.11.11`, pilote NVIDIA compatible avec la roue CUDA, accès au **checkpoint complet** (poids, tokenizer, chat template, `config.json`, `rswa_training.json`) et une image de test. Le checkpoint n'est pas fourni par ce dépôt. Windows sert aux tests CPU et à la démo, pas à l'installation vLLM de ce projet.

Depuis la racine, avec le service arrêté si l'on intervient sur son venv :

```bash
git pull --ff-only
uv sync
.venv/bin/python src/inference/rswa_patch.py --apply --variant pruned --mtp3
.venv/bin/python src/inference/rswa_patch.py --mtp3
uv run --no-sync --with pytest pytest tests/test_rswa_kernel.py -v
```

La deuxième commande de patch **vérifie sans écrire** et doit annoncer `RSWA pruned patch verified for vLLM 0.34.0 (with MTP3)`. `pruned` est le noyau choisi pour reproduire la machine testée, pas un état imposé par `main.py`. Sur une roue vierge, `--apply --mtp3` installe `baseline` si `--variant` n'est pas précisé ; sur une roue déjà patchée, il conserve sa variante.

Valider ensuite l'inférence réelle avec une A40 libre :

```bash
HTR_MTP_TOKENS=3 \
HTR_RSWA_TEST_MODEL=/home/fizainef/Weights/Qwen3.5-4B-06112026-sft-html-convert-rswa-w512_grpo \
uv run --no-sync --with pytest pytest tests/test_rswa_checkpoint.py -v -s \
  --image-path /chemin/vers/page.jpg
```

Chercher `RSWA ready` (dont `mtp_tokens: 3`), `RSWA target ran`, `RSWA draft ran` et `RSWA first KV eviction`. Le test force `window + 3 × block_size` tokens avec EOS ignoré : avec W=512 et bloc 576, cela fait 2 240 tokens. Il vérifie la longueur et la finitude des log-probabilités, pas la qualité de la transcription. `HTR_RSWA_TEST_OUTPUT=/chemin/sortie.txt` conserve la sortie brute, qui peut être répétitive après la fin naturelle du document.

Lancer ensuite **soit** le service **soit** le processus au premier plan, jamais les deux :

```bash
sudo systemctl start lel
journalctl -u lel -n 100 --no-pager
```

```bash
# Variante au premier plan, uniquement si lel est arrêté
.venv/bin/python src/main.py
```

**Ne pas lancer `src/main.py` avec `uv run`.** Ray récent peut reproduire `uv run` dans ses workers, depuis un autre répertoire, et créer un venv vide (`No module named 'ray'`). Les commandes de test qui n'initialisent pas un cluster Ray peuvent passer par `uv run`. L'unité `lel` doit utiliser directement le Python du venv ; voir [l'exemple systemd](../README.md#service-systemd-etcsystemdsystemlelservice). `systemctl cat lel` montre l'unité et ses surcharges. Pour une modification durable, `sudo systemctl edit lel`, par exemple :

```ini
[Service]
Environment="HTR_MODEL_PATH=/chemin/absolu/checkpoint"
Environment="HTR_MTP_TOKENS=1"
```

Ces valeurs sont un **exemple de surcharge**, pas les valeurs à installer pour reproduire MTP3. Redémarrer après modification. Éviter de modifier le chemin du modèle dans un commit local propre au serveur : la variable d'environnement évite une divergence au prochain pull.

### Dépendances et traçabilité

[`pyproject.toml`](../pyproject.toml) fixe `vllm==0.34.0`, `torch==2.14.0+cu130`, `torchvision==0.29.0+cu130`, `torchaudio==2.11.0+cu130`, et `transformers>=5.16.1,<5.23.0`. La référence A40 rapportée utilisait Transformers `5.22.2`. L'index `pytorch-cu130` est explicite pour les trois paquets PyTorch. Python, la cible Linux et les options de build **vLLM** existantes ont été conservés. On utilise une roue précompilée ; les anciens drapeaux `sm_80` ne définissent pas l'architecture de l'A40 (`sm_86`) ni celle d'une roue déjà compilée.

Les overrides `uv` de la version précédente ont été retirés : ils doublaient ou contournaient les contraintes. Les installations directes de `flash-attn`, `flash-linear-attention` et `causal_conv1d` ainsi que leurs réglages de build ont été retirées : le chemin Qwen3.5 utilise les implémentations livrées par vLLM. Cela évite les compilations externes inutiles. En vLLM 0.34.0 l'extension CUDA testée est `vllm._C_stable_libtorch`, pas `vllm._C`.

Le `uv.lock` est ignoré par Git et Ray n'est pas figé. `uv sync` reproduit les contraintes, **pas** à lui seul toutes les versions transitives de l'A40. Pour archiver un essai, garder son `uv.lock` local, le rapport JSON et :

```bash
git rev-parse HEAD
nvidia-smi
uv pip freeze --python .venv/bin/python > versions-a40.txt
.venv/bin/python src/inference/rswa_patch.py --mtp3
```

## 3. RSWA : contrat, patchs, démarrage et vérifications

### Masque et cache

Pour une requête, `P` est le nombre de tokens du prompt **après expansion multimodale**, image comprise. Il ne s'agit ni de la longueur du texte du prompt ni de la longueur du chunk en cours. `W` vaut 512 dans le checkpoint testé, mais provient du manifeste :

```text
visible(q, k) = k <= q ET (k < P OU q - k < W)
```

Le préfixe est causal et reste visible. La génération voit le préfixe et sa fenêtre de réponse, token courant inclus. Les positions logiques/MRoPE ne sont ni renumérotées ni compactées. Le `RSWAManager` natif libère uniquement les blocs **entiers** dans la zone invisible ; les positions correspondantes sont remplacées par le bloc nul. `W` n'est donc pas une taille exacte de cache, notamment avec les blocs de 576 tokens observés et les états GDN.

[`src/inference/rswa_policy.py`](../src/inference/rswa_policy.py) détecte le manifeste **avant les imports vLLM**. Il vérifie `allow_rswa: true`, `version: 1`, `policy: "causal_prefix_response_window"`, `native_vllm_autoload: false`, un entier positif `rswa_window` et `backend_train` `flex` ou `sdpa_dense`. Les métadonnées du manifeste doivent correspondre à `config.json["rswa_training"]` et `config.json["text_config"]["rswa_training"]` ; seul `serving_requirement` est retiré de la comparaison côté manifeste. Les champs `rswa_window` non encapsulés à la racine du config ou du `text_config` sont interdits. L'override HF injecte la fenêtre **en mémoire** pour target et draft, sans renommer les architectures vLLM et sans toucher aux poids, au tokenizer ou aux fichiers du modèle.

Exemple de manifeste issu de l'export testé (**pas** un moyen de convertir un checkpoint standard en modèle entraîné avec RSWA) :

```json
{
  "allow_rswa": true,
  "rswa_window": 512,
  "policy": "causal_prefix_response_window",
  "version": 1,
  "backend_train": "flex",
  "native_vllm_autoload": false,
  "serving_requirement": "RSWA cache manager + attention + multimodal MTP draft prefix support"
}
```

Le contrat accepte le Qwen3.5 **dense multimodal non quantifié**, une couche MTP native, BF16, TP/DP/PP et parallélismes de contexte à 1, exécuteur local `mp` avec `spawn`, `TRITON_ATTN` sur target et draft, état récurrent GDN FP32, `TORCH_SDPA` pour la vision, mode eager et scheduling synchrone. Pas de prefix cache, pas de cache processeur multimodal, pas de MTP parallèle, dynamique, adaptatif ou multi-module. Les réglages réservés ne sont pas silencieusement écrasables.

### Fichiers modifiés dans la roue vLLM

Le script [`src/inference/rswa_patch.py`](../src/inference/rswa_patch.py) est la source des modifications. Il valide `0.34.0` et les SHA256 avant toute écriture ; il prépare toutes les modifications puis remplace chaque fichier par un fichier temporaire sur le même disque. Le hash après modification est vérifié. Une source inconnue fait échouer la commande, même si le numéro de version est bon. L'installation est idempotente. **Ne pas remplacer ces empreintes pour “faire passer” une roue différente** : comparer les sources et revoir chaque ancrage, puis revalider sur l'A40.

| Chemin sous `site-packages/vllm/` | Modification appliquée |
|---|---|
| `model_executor/models/qwen3_next.py` | Lorsqu'un `ModelConfig` porte la fenêtre RSWA, instancie `RSWAAttention` au lieu de `Attention` pour le décodeur causal sans dual-chunk. Le constructeur est partagé par Qwen3.5 et son draft. C'est ce qui demande `RSWASpec` et déclenche le vrai gestionnaire d'éviction. |
| `v1/attention/ops/triton_unified_attention.py` | `baseline` met à zéro les valeurs V invisibles pour toutes les lignes valides de la tuile avant `P @ V` : `0 * NaN` peut sinon empoisonner l'output avec du cache recyclé. `pruned` ajoute le saut des tuiles totalement dans le gap ou le futur causal ; il garde les tuiles aux frontières et l'ordre des tuiles conservées. Les segments split-K vides écrivent l'état neutre de softmax. |
| `v1/worker/gpu/spec_decode/autoregressive/speculator.py` (`--mtp3`) | Transmet `input_batch.prompt_lens`, aligné sur les requêtes actives, au chemin de décodage multi-étapes. |
| `v1/worker/gpu/spec_decode/speculator.py` (`--mtp3`) | Fait suivre `rswa_prefix_lens` jusqu'à `build_attn_metadata`. Sans cela, la reconstruction des métadonnées des étapes 2 et 3 du draft perd le préfixe et Triton ne peut pas appliquer RSWA correctement. |

MTP3 signifie trois tokens spéculatifs **séquentiels** sur la même couche MTP, pas trois couches ni trois workers. Suivant les acceptations, une mise à jour peut transporter plusieurs tokens ; les routes doivent les traiter tous. `K=3` s'applique au moteur entier, donc aussi au TSV et au HTML non streamé. Avec `K=1`, les deux fichiers MTP3 peuvent rester patchés, mais le chemin multi-étapes n'est pas utilisé.

### Du lancement à la première requête

1. [`src/main.py`](../src/main.py) démarre Ray et construit `HTREngine` dans le constructeur **async** du réplica. On attend `verify_ready()` avant de servir ; à l'arrêt, le moteur vLLM est fermé.
2. [`src/inference/engine.py`](../src/inference/engine.py) découvre le checkpoint, construit les options et vérifie les patchs **avant** de lancer vLLM. La branche RSWA vérifie la configuration effective produite par `AsyncEngineArgs`; la branche standard garde `from_engine_args`. Les imports liés à vLLM et au préprocesseur restent locaux aux méthodes qui les utilisent.
3. [`src/inference/rswa_runtime.py`](../src/inference/rswa_runtime.py) vérifie les groupes `RSWAManager`/GDN du scheduler. Le RPC `rswa_inspect` vérifie sur le GPU les couches target et draft, les caches BF16 alloués, les états GDN et le nombre de blocs spéculatifs. Avec MTP3 on exige la reconstruction des métadonnées et non la mise à jour fusionnée en place.
4. Après attestation, chaque passage Triton RSWA vérifie notamment la présence du préfixe, le bon `W`, la causalité et l'absence d'un préfixe bidirectionnel. En cas d'échec d'attestation, le réplica s'arrête au lieu de servir en attention complète. Le scheduler journalise le premier vrai remplacement d'un bloc par le bloc nul ; les messages target/draft confirment le premier forward protégé, pas un benchmarking de chaque token.

### Valeurs modifiables au démarrage

| Réglage | Défaut | Pour changer |
|---|---|---|
| `HTR_MODEL_PATH` | Chemin du checkpoint RSWA du tableau §1 | Chemin absolu dans l'environnement de `lel` ou dans la commande de lancement. Ne pas éditer les poids pour activer RSWA. |
| `HTR_REQUIRE_RSWA` | Obligatoire, sauf valeur exacte `0` | `0` accepte un modèle standard ; ne désactive pas RSWA quand le manifeste est présent et valide, et ne rend pas acceptable un manifeste cassé. |
| `HTR_MTP_TOKENS` | `3` dans `main.py` | `1` revient au chemin court RSWA. Les valeurs intermédiaires ne sont pas supportées. Sur modèle standard, le réglage historique par nom Qwen3.5 reste `K=2`. |
| `HTR_ENABLE_THINKING` | Faux | `1` transmet `enable_thinking=True` au chat template Qwen3.5. Un template personnalisé doit reconnaître cette option. |
| Taille de contexte | `HTML_CONTEXT_LENGTH = 24576` | Valeur commune dans [`html_output.py`](../src/processing/html_output.py) et `main.py`. Garder de la place pour le prompt texte et image **en plus** de `HTML_MAX_TOKENS`. La route HTML refuse une config dont le contexte est inférieur à cette valeur (`503`). |
| CUDA et concurrence | Réglages de `main.py` (§1) | Modifier les arguments Ray/vLLM dans `main.py`, puis mesurer sous charge. `max_num_batched_tokens` est un budget de travail par itération, pas le nombre maximal de tokens de sortie d'une requête. |

`HTREngine` et les tests/scripts directs gardent leurs propres valeurs par défaut : en particulier, le smoke test utilise MTP1 sauf `HTR_MTP_TOKENS=3`. Le benchmark existant fixe MTP1 et ne lit pas cette variable.

## 4. Transcription et formats

### Prétraitement et TSV historique

[`src/processing/preprocessing.py`](../src/processing/preprocessing.py) conserve le découpage des images paysage en deux zones qui se chevauchent et l'appel au chat template. Le seul nouveau paramètre est `enable_thinking`, passé uniquement pour Qwen3.5. En mode sans thinking, le template officiel Qwen3.5 insère un bloc `<think>` déjà fermé ; si une exportation ignore l'option, vérifier son `chat_template.jinja`. Le chevauchement peut produire deux lectures de contenu proche de la séparation ; il n'y a pas de déduplication côté serveur.

`POST /transcribe` (JSON) et `POST /transcribe/stream` (NDJSON) conservent le multipart `image`, la conversion en polygones et les prompts de [`templates/tasks/HTR.py`](../src/templates/tasks/HTR.py). Le budget demandé par le chemin TSV reste 12 288 tokens ; le contexte total du serveur a changé. Le parseur [`postprocessing.py`](../src/processing/postprocessing.py) n'a pas été modifié.

Le correctif [`stream_metrics.py`](../src/processing/stream_metrics.py) compte le log-prob du token **effectivement choisi**, par ID, pas la première entrée du dictionnaire ni seulement le dernier token d'une mise à jour. Les routes partagent des deltas vLLM (`RequestOutputKind.DELTA`). Un token de plusieurs caractères est réparti en proportion s'il recouvre une frontière de ligne ou de label. La perplexité est `exp(-somme_logprobs / poids)` ; le document utilise tous les tokens émis, la métrique texte se limite aux labels reconnus. Si les chaînes de tokens ne s'alignent pas sur le texte incrémental, le texte réel est conservé et la moyenne du delta sert de repli. Ce correctif élimine le biais systématique du regroupement des tokens ; il ne rend pas deux inférences GPU distinctes bit à bit identiques.

### HTML complet et streaming

Le prompt [`src/templates/tasks/HTML.py`](../src/templates/tasks/HTML.py) reproduit l'entrée `html` du fichier de travail `structured_html.json` fourni. Les autres tâches de ce JSON ne sont pas implémentées ici. Le modèle reçoit `[FORMAT:html]` et doit préserver le tableau, les cellules vides, les titres, les notes, les signatures et les `coord`. Ce chemin ne passe ni par le regex TSV, ni par le parseur de polygones, ni par le calcul des perplexités. `HTML_MAX_TOKENS = 16_384` est un maximum **par fragment**, pas une longueur forcée. La génération peut s'arrêter normalement avant ; `"length"` peut indiquer un HTML tronqué.

`POST /transcribe/html` produit un JSON final. `pages` reste dans l'ordre gauche → droite, même si la génération droite termine en premier. Chaque page contient `html`, `offsetX`, `width`, `height` et `finishReason`. Seules les clôtures Markdown externes sont retirées. Aucune correction, validation sémantique ou désinfection du HTML brut n'est effectuée par l'API.

`POST /transcribe/html/stream` produit une ligne NDJSON par événement :

```jsonl
{"type":"metadata","imageWidth":4991,"imageHeight":3871,"model":"Qwen3.5-4B","pages":[{"pageIndex":0,"offsetX":0,"width":2570,"height":3871},{"pageIndex":1,"offsetX":2420,"width":2571,"height":3871}]}
{"type":"delta","pageIndex":0,"html":"```html\n<table><tr>"}
{"type":"delta","pageIndex":1,"html":"<p>Début de la deuxième page</p>"}
{"type":"done","pageIndex":1,"finishReason":"stop"}
{"type":"error","pageIndex":0,"message":"HTML generation failed"}
{"type":"complete"}
```

`pageIndex` démarre à zéro et indexe le fragment de l'image, **pas** le numéro imprimé `<page_number>` dans le document. Un `delta` peut couper une balise, un attribut ou du texte : il faut concaténer par page, dans l'ordre de réception sur cette page. `error` termine un fragment en erreur sans invalider automatiquement les autres. `complete` marque la fin du transport, pas la réussite de tous les fragments. La queue serveur est bornée ; la fermeture de l'itérateur annule les tâches de génération encore actives. Le flux demande `X-Accel-Buffering: no` au proxy, sans pouvoir forcer la politique d'un proxy extérieur.

Pour projeter une boîte sur le scan : `x = offsetX + x_normalisé * width / 1000` et `y = y_normalisé * height / 1000`. Appliquer la même conversion aux coins maximum. Le visualiseur dessine des rectangles alignés sur les axes ; il affiche l'angle en info-bulle, mais ne fait pas pivoter les rectangles.

## 5. Démo, fichiers à éditer et régénération

| Fichier | Rôle / modifications utiles |
|---|---|
| [`scripts/demo_transcription_html.html`](../scripts/demo_transcription_html.html) | Interface commune : import/dépôt, démarrage automatique **après sélection** (aucun envoi à l'ouverture sans image), lecture NDJSON, état compact et téléchargement. Pour modifier les boutons, les messages ou la barre, éditer ce fichier. |
| [`scripts/demo_transcription_html.py`](../scripts/demo_transcription_html.py) | Serveur local stdlib sur `127.0.0.1:8765`, `--port`, `--no-browser`. Reçoit le fichier du navigateur, vérifie origine locale, format et taille (JPEG/PNG/WebP, 30 Mio), crée le multipart et appelle l'URL HTTPS fixe. Adapte le NDJSON en mises à jour du visualiseur. Pas de dépendance CUDA, Ray, Python tiers ni CORS public pour ce mode. |
| [`scripts/preview_transcription_html.py`](../scripts/preview_transcription_html.py) | Rendu des aperçus : liste blanche HTML, conservation de `rowspan`/`colspan`, projection des `coord`, SVG, tableau/notes/signatures/numéros de pages, overlay, sélection croisée et zoom relatif au fit de l'image entière. Produit également l'aperçu autonome en CLI. |
| [`scripts/portable_transcription_html.js`](../scripts/portable_transcription_html.js) | Adaptateur navigateur : multipart `FormData`, aucun credential, état par `pageIndex`, rendu progressif et liste blanche DOM. Le HTML partiel est analysé dans un `<template>` inerte ; le navigateur ferme provisoirement le tableau, sans modifier la chaîne reçue. |
| [`scripts/build_portable_demo.py`](../scripts/build_portable_demo.py) | Assemble l'interface, le JS portable et le visualiseur dans un fichier HTML sans dépendance externe. À relancer après modification d'un des trois fichiers sources du rendu. |
| [`demo-lettres-en-lumieres.html`](../demo-lettres-en-lumieres.html) | **Fichier généré**, servi par `GET /demo`, aussi partageable seul. Ne pas le maintenir à la main. La copie générée doit être committée avec ses sources. |
| [`src/api/demo_cors.py`](../src/api/demo_cors.py) | Ajoute `Access-Control-Allow-Origin: null` et `Vary: Origin` uniquement pour le `POST /transcribe/html/stream` concerné. Pas de credentials, pas de CORS global. Utile pour un `file://` si le proxy le permet ; `null` désigne aussi d'autres origines opaques, ce n'est pas une authentification. |

Le relais et le client portable ne rendent pas directement le HTML donné par le modèle : ils reconstruisent les éléments autorisés, enlèvent notamment scripts, gestionnaires d'événements et attributs de présentation non pris en charge. Les nouveaux tags/attributs doivent être vérifiés dans **les deux listes blanches**. Pour afficher le flux, un aperçu est mis à jour environ toutes les 150 ms par page, puis une fois à sa fin. L'iframe est sandboxée ; le visualiseur et ses exports ont une CSP qui autorise le script interactif généré et l'image intégrée. Le rendu garde l'image originale à côté du HTML ; les segments et le zoom restent utilisables pendant le streaming.

```bash
# Client local (démarre un relais et ouvre le navigateur)
python scripts/demo_transcription_html.py

# Régénérer la version distribuable et hébergée
python scripts/build_portable_demo.py

# Recréer un aperçu hors ligne à partir d'une réponse JSON déjà enregistrée
python scripts/preview_transcription_html.py transcription-html.json page.jpg transcription-apercu.html
```

Modes à distinguer :

- **Public recommandé** : `https://lel.cotedor.fr/demo` → `/transcribe/html/stream` sur la même origine. Pas de CORS nécessaire ; le proxy doit toutefois transmettre `/demo` et la route d'API.
- **Fichier HTML seul** : ouverture depuis le disque, appel HTTPS avec `Origin: null`. Fonctionne seulement si le proxy laisse passer cet échange ; sinon utiliser le lien **Ouvrir la démo en ligne**. L'accès Internet reste nécessaire pour une nouvelle inférence.
- **Aperçu téléchargé** : image et résultats déjà intégrés ; visualisation hors ligne, sans nouveau calcul et sans Python.

Pour changer de serveur public, modifier l'adresse dans `demo_transcription_html.py`, `portable_transcription_html.js`, le texte/lien dans `demo_transcription_html.html` et la liste CSP du script de build, puis régénérer. Pour agrandir le délai d'attente, distinguer le timeout du relais (600 s), celui du navigateur/client et le timeout du proxy extérieur. Augmenter l'un ne change pas les autres.

## 6. Vérification HTTP, benchmark et limites connues

Avec le serveur lancé, depuis le serveur lui-même :

```bash
curl -fsS http://127.0.0.1:8000/model_info
curl --fail-with-body -sS -F 'image=@../FRAD021_10M106_05_0004.jpg' \
  http://127.0.0.1:8000/transcribe/html -o transcription-html.json
curl -N --fail-with-body -sS -F 'image=@../FRAD021_10M106_05_0004.jpg' \
  http://127.0.0.1:8000/transcribe/html/stream
uv run --no-sync --with pytest pytest tests/test_transcribe.py -v \
  --base-url http://127.0.0.1:8000 --image-path ../FRAD021_B_012126_00013.jpg
```

Comparer le flux local avec `https://lel.cotedor.fr/transcribe/html/stream` pour isoler le reverse proxy. `curl -N` ne tamponne pas le flux côté client. Un HTTP 200 accompagné d'un texte de maintenance et de `Retry-After` ne représente pas une transcription réussie. `GET /health` confirme l'accès HTTP ; pour vérifier le vrai modèle, lire `/model_info` et `RSWA ready` dans les logs.

[`scripts/benchmark_rswa.py`](../scripts/benchmark_rswa.py) mesure le moteur directement, hors HTTP, prétraitement initial et rendu. Il échauffe les formes utilisées, exécute chaque crop à chaque concurrence demandée et écrit le JSON après chaque cas. Le cas normal garde EOS ; les longueurs contrôlées imposent `min_tokens=max_tokens` et ignorent EOS. Les débits dépendent des paramètres du moteur et du type de tâche ; ce script utilise les **prompts et contraintes TSV**, MTP1, thinking désactivé. Il ne mesure pas le temps d'affichage HTML ni le bénéfice de MTP3 sous charge. Ne pas comparer un test HTML/MTP3 à cette référence comme s'il ne différait que par le patch Triton.

```bash
# Serveur arrêté ; même roue, même checkpoint, mêmes images et options pour les deux variantes
.venv/bin/python src/inference/rswa_patch.py --apply --variant baseline --mtp3
.venv/bin/python scripts/benchmark_rswa.py \
  --model /home/fizainef/Weights/Qwen3.5-4B-06112026-sft-html-convert-rswa-w512_grpo \
  --normal-image ../FRAD021_B_012126_00013.jpg \
  --long-image ../FRAD021_10M106_05_0004.jpg \
  --long-lengths 8192 16384 --concurrency 1 --label baseline --output rswa-baseline.json
```

Rejouer après `--apply --variant pruned --mtp3` avec `--label pruned --output rswa-pruned.json`. Contrôler les SHA256 du rapport en plus de son label. Sur l'A40, les mesures fournies à concurrence 1 et une répétition étaient **83,44 → 83,69 tok/s** à 8k et **82,93 → 83,44 tok/s** à 16k, pic KV **0,79 %** dans les deux variantes à 16k. Ce n'est pas une preuve de gain reproductible sous charge. Pour la suite : plusieurs répétitions, concurrences adaptées, conserver les mêmes réglages et isoler séparément variante Triton, K et type de sortie. Le `Maximum concurrency` calculé au démarrage est une estimation de capacité, pas un résultat de charge.

### Quels tests relancer ?

| Fichier | Ce qu'il vérifie / prérequis |
|---|---|
| [`test_rswa_policy.py`](../tests/test_rswa_policy.py) | Contrat manifeste/config, modifications HF en mémoire, refus des options incompatibles, K1/K3. CPU. |
| [`test_rswa_kernel.py`](../tests/test_rswa_kernel.py) | Bornes du masque, intervalle de tuiles `pruned` (CPU) ; CUDA/BF16, extension vLLM, prefill/décodage, référence FP32, NaN en gap et décodage 16k avec et sans split-K (Linux + GPU). |
| [`test_rswa_checkpoint.py`](../tests/test_rswa_checkpoint.py) | Vrai modèle multimodal, attestation, génération longue et log-probs finies. Linux + GPU libre, `HTR_RSWA_TEST_MODEL`, image de test et `HTR_MTP_TOKENS=3` pour MTP3. |
| [`test_benchmark_rswa.py`](../tests/test_benchmark_rswa.py) | Comptabilité, parallélisme simulé et détection d'une longueur forcée non atteinte. CPU. |
| [`test_transcription_metrics.py`](../tests/test_transcription_metrics.py) | Perplexités indépendantes du regroupement standard des deltas ; sélection du token émis, fin de ligne, EOS. CPU. |
| [`test_html_output.py`](../tests/test_html_output.py) | Clôtures Markdown, budget, ordre des crops, multiplexage et annulation du flux. CPU, moteur simulé. |
| [`test_demo_transcription_html.py`](../tests/test_demo_transcription_html.py) | Relais multipart, tableau partiel, filtre HTML, types des segments et aperçu. CPU, API simulée. |
| [`test_demo_cors.py`](../tests/test_demo_cors.py) | CORS opaque limité au flux HTML, sans credentials. CPU. |
| [`test_transcribe.py`](../tests/test_transcribe.py) | Suite HTTP TSV existante (non modifiée) ; nécessite le serveur démarré et une vraie image. |

Exemple Windows qui ne résout pas le projet Linux :

```powershell
uv run --no-project --python 3.11.11 --with pytest --with requests --with pillow --with numpy --with shapely --with fastapi --with httpx2 pytest tests/test_rswa_policy.py tests/test_rswa_kernel.py tests/test_rswa_checkpoint.py tests/test_transcription_metrics.py tests/test_html_output.py tests/test_demo_transcription_html.py tests/test_demo_cors.py tests/test_benchmark_rswa.py -q
```

Sur Windows, les tests CUDA et checkpoint sont ignorés ; cela ne certifie pas la machine cible. La recette manuelle du navigateur : import et glisser-déposer, aucun bouton de confirmation, apparition d'un tableau avant sa fermeture, deux `pageIndex` indépendants, zoom et sélection croisée, téléchargement puis ouverture hors ligne. La qualité de lecture des manuscrits et la fidélité des colonnes sont à comparer visuellement ; des attributs valides ne garantissent pas une bonne transcription.

## 7. Revenir en arrière et diagnostiquer

| Symptôme / besoin | Action |
|---|---|
| MTP3 pose problème, conserver RSWA et le streaming | Définir `HTR_MTP_TOKENS=1` dans `lel` et redémarrer. Les fichiers MTP3 peuvent rester patchés. |
| Annuler seulement le pruning | Arrêter le serveur ; `.venv/bin/python src/inference/rswa_patch.py --apply --variant baseline --mtp3` ; redémarrer. Le masque et l'éviction restent actifs. |
| Enlever les deux patchs MTP3 de la roue | Arrêter le serveur ; `uv sync --reinstall-package vllm` ; réappliquer `--apply --variant pruned` **sans** `--mtp3` ; configurer `HTR_MTP_TOKENS=1` ; redémarrer. Choisir `baseline` à la place de `pruned` si souhaité. |
| `RSWA patch missing`, hash inattendu | Vérifier `ExecStart` et le venv, la version de la roue et si un `uv sync`/une réinstallation l'a remplacée. Ne pas contourner l'attestation. |
| `No module named 'ray'` dans `raylet` | Lancer avec `.venv/bin/python` directement ; vérifier le Python effectif de l'unité. |
| La route ou `/demo` répond 404 uniquement via HTTPS | Tester d'abord sur `127.0.0.1:8000` ; contrôler ce que le proxy route. Rien dans le dépôt ne change la configuration du proxy. |
| Fichier HTML local bloqué par CORS | Utiliser `https://lel.cotedor.fr/demo` ; si l'accès HTTPS est impossible, le relais Python local ou un aperçu déjà téléchargé sont deux alternatives distinctes. |
| HTML reçu seulement à la fin | Comparer `curl -N` direct et public ; vérifier buffering/compression/timeout du proxy. Les clôtures synthétiques ne changent pas le transport. |
| Le service répond mais sur le mauvais modèle | Vérifier `HTR_MODEL_PATH` dans `systemctl cat lel` et `/model_info`. Un environnement systemd peut surcharger `main.py`. |
| Retour complet au code ancien | Tester `e9c81e5` dans un **worktree séparé**, avec son propre venv et l'ancien checkpoint, avant de basculer le service. Ne pas mélanger anciennes dépendances, code nouveau et patchs d'une autre roue. |

Les rapports `rswa-baseline*.json`, images, prompts de référence et sorties du travail local ne sont pas des dépendances de déploiement. Ne pas les ajouter au dépôt par inadvertance. Le brief d'adaptation vLLM 0.28 et le `policy.py` de référence expliquent le contrat initial, mais **ne sont pas importés** par ce serveur 0.34.

Pour consulter ou tester le code d'origine sans modifier la branche ou le venv du service :

```bash
git worktree add --detach ../LeLRay-avant e9c81e5
```

Ce dossier ne récupère ni l'ancien venv ni le checkpoint. Pour une restauration exacte de la machine d'origine, utiliser une sauvegarde de son environnement ; le commit seul ne suffit pas.

## 8. Historique des commits

| Commit | Diff à lire pour comprendre… |
|---|---|
| `af78f1a` | Migration vLLM, PyTorch et contraintes Transformers. |
| `42265d6` | Tests initiaux CPU/CUDA et oracle RSWA. |
| `e770b95` | Suppression des compilations externes devenues redondantes. |
| `c30d01a` | Import de la vraie extension CUDA vLLM 0.34. |
| `2847558` | Patch audité de la roue, raccord Qwen3.5 et valeurs V invisibles. |
| `6001678` | Détection du checkpoint, attestation Ray/worker et test multimodal long. |
| `6b47d4b` | Logs RSWA dans les sous-processus et sauvegarde optionnelle de génération. |
| `1be5bc2` | Démarrage Ray avec le Python du venv, sans relance `uv run`. |
| `f0eb683` | Correction des imports paresseux qui provoquaient des HTTP 500. |
| `df079d9` | Thinking désactivé par défaut. |
| `c5f440f` | Perplexités TSV calculées par token plutôt que par delta. |
| `dc824a3` | Benchmark multi-longueurs et métriques de référence. |
| `474b978` | Variante Triton `pruned`, retour à `baseline` et tests split-K/16k. |
| `23fd5f7` | Route HTML complète, prompt fourni et budget 16k. |
| `df676fa` | Démo navigateur avec relais Python et sauvegarde autonome. |
| `7440d3a` | Checkpoint RSWA par défaut dans `main.py`. |
| `2374edb` | Flux HTML multipage, rendu progressif et patch séquentiel MTP3. |
| `0976d2c` | MTP3 activé par défaut au démarrage Ray Serve. |
| `cd41ffc` | Interface compacte, fit image et couleurs sémantiques. |
| `fabecef` | Import déclenchant la transcription et fichier portable avec CORS ciblé. |
| `f5447e9` | Démo publique même origine, indépendante de CORS. |
