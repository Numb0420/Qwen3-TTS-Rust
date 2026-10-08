# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

## Déploiement actuel

Service Qwen3-TTS en Rust pour Windows : **Qwen3-TTS-12Hz-1.7B-Base, Q5_K_M, un seul moteur**. Talker et Predictor utilisent llama.cpp Vulkan ; le vocodeur utilise ONNX Runtime sur CPU. Validation sur **AMD Radeon 8060S**. Les chiffres concernent ce projet uniquement ; CUDA, Metal et les autres machines ne sont pas évalués ici. Mise à jour : 2026-10-08.

## Fonctions disponibles

- WebUI : génération complète, streaming WebSocket, arrêt et délais jusqu’au premier son.
- Import WAV avec transcription exacte, extraction ICL complète, chargement après redémarrage.
- Post-traitement séparé de vitesse/volume (0–100), désactivé à 50.
- `<phoneme py="lù">六</phoneme>` devient du pinyin suivi d’un espace avant inférence ; ce n’est pas du SSML natif.
- Lanceur Windows avec icône de notification, IP/port et mode CPU/Vulkan configurables, scripts de packaging.
- Voix déployées : 39 (par défaut), xiaolu, xiaoyan. Fichiers exclus de Git ; le service Base n’utilise pas les identifiants CustomVoice Serena/Vivian.

## Mesures du premier paquet

**Windows / Radeon 8060S / Base 1.7B Q5 / Vulkan + vocodeur CPU**. Actualités chinoises, instruction vide ; Temperature 0.7, Top-K 50, Top-P 1, graine 43, vitesse/volume 50. Quatre requêtes complètes après une première synthèse. Mesure de l’envoi sur WebSocket déjà connecté au premier PCM cible, distincte du premier son du navigateur et du temps de synthèse total.

| Lancement / voix | Moyenne | Plage |
| --- | ---: | ---: |
| Lancement local normal / 39 | 322.47 ms | 313.31–333.28 ms |
| Lanceur portable / 39 | 312.46 ms | 310.76–314.31 ms |

```text
他也提到，去年河北十个市，县，获评第七届全国文明城市，数量是前六届的总和，全国双拥模范城，实现设区市全覆盖。人民群众的安全得到切实维护。
```

Autre test du même jour : 39 337.61 ms, xiaolu 319.95 ms, xiaoyan 328.89 ms (trois essais chacun, texte antérieur). Ne pas soustraire ces tests pour calculer un gain. Les 20 sorties PCM complètes du comparatif portable sont identiques octet par octet ; cette preuve concerne les versions et entrées testées.

Le texte, la voix, les files d’attente et la charge font varier le délai. Un utilisateur a mesuré 411 ms au paquet et 431 ms au son ; non reproduit via API, cause inconnue. La première requête locale après redémarrage a pris environ 2.47 s, hors tableau. Pas de synthèse automatique de préchauffage. **Un délai systématiquement ≤300 ms n’est pas démontré.** Optimisations : projections fixes, plage d’échantillonnage, envoi anticipé de quatre trames, mise à jour des états sans onde de référence et réduction du calcul d’onde au second appel. Les nouvelles voix doivent passer la calibration au démarrage. Voir section 34 et comparaison du packaging.

## Exécution et configuration

Extraire le paquet portable complet et lancer `启动TTS.exe`. Le menu ouvre la page, les réglages, les journaux ou arrête le service. Les utilisateurs n’ont pas besoin des outils Rust/Python/Visual Studio ; Vulkan exige un pilote graphique fonctionnel. Le développement et le packaging nécessitent les outils de compilation. Redémarrer après modification :

`server-config.json`

```json
{
  "inference_device": "vulkan",
  "host": "0.0.0.0",
  "port": 3000,
  "speakers": ["39", "xiaolu", "xiaoyan"],
  "open_browser": true
}
```

[http://127.0.0.1:3000/](http://127.0.0.1:3000/)

**Le dépôt source n’est pas une distribution portable complète.** Modèles, runtime, EXE et JSON de voix sont exclus. Certains modèles convertis, ONNX optimisés et backends n’ont pas encore de téléchargement public. Préparer les fichiers correspondants et le répertoire de publication compilé et vérifié avant de lancer :

```powershell
pwsh -NoProfile -File .\start-webui.ps1
```

## Voix personnalisées

WAV clair, un seul locuteur, sans musique, avec transcription exacte. Import Web : 1–30 s ; conversion à 24 kHz sans modifier l’original. Full conserve codes de référence, texte et vecteur ; Xvec conserve uniquement le vecteur et n’équivaut pas à ICL complet. Redémarrer après import ; ajouter les JSON manuels avec `-BaseSpeakers`.

```powershell
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice -Mode Full -RefTextFile .\reference.txt
.\create-speaker.ps1 -AudioPath .\reference.wav -Name my-voice-xvec -Mode Xvec
pwsh -NoProfile -File .\start-webui.ps1 -BaseSpeakers my-voice
```

## API

Même port HTTP/WebSocket (3000 par défaut). `POST /api/tts` : JSON avec WAV Base64 ; `GET /api/speakers` : voix ; `GET /health` : état. Envoyer ce JSON à `ws://127.0.0.1:3000/api/tts/stream`. Préciser les paramètres pour reproduire le WebUI ; Temperature omis vaut 0 côté serveur.

```json
{
  "text": "你好，今天天气很好。",
  "speaker": "39",
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 1,
  "seed": 43,
  "instruction": "",
  "speed": 50,
  "volume": 50
}
```

Messages binaires : PCM mono 24 kHz Float32 little-endian. `segment_done` termine un segment ; le message texte `end` termine l’entrée, suivi de `done`. Fermer le WebSocket pour annuler ; `end` n’annule pas et aucun endpoint HTTP d’arrêt distinct n’existe. Finir de lire l’audio reçu après `done`. À Temperature 0, Talker choisit le maximum : ses Top-K/Top-P ne s’appliquent pas. Predictor garde ses réglages séparés (0.9 / 50 / 1).

## Documentation et origines

Documents détaillés actuellement en chinois. Projet issu de l’implémentation Rust amont, utilisant Qwen3-TTS, llama.cpp et ONNX Runtime ; conserver les licences et attributions des composants.

- [Base 1.7B — models / 模型](../models/base-1.7b/README.md)
- [Configuration / 配置](../server-config说明.md)
- [Portable package / 绿色版](../绿色版打包说明.md)
- [Optimization measurements / 优化方法](../优化方法.md)
- [Preserved baseline / 基线](../基线版本.md)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
## Licences et remerciements

Ce projet est développé à partir de [cgisky1980/Qwen3-TTS-Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust). Merci à son auteur pour l’implémentation Rust et les fondations du projet.

- [Component licenses / 组件许可证](../tools/portable-licenses/)
