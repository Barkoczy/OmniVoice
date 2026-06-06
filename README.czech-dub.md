# tts-service — český dabing a TTS přes OmniVoice

Lokální služba pro převod textu na řeč a **dabing videí v češtině** postavená na
modelu [OmniVoice](https://github.com/k2-fsa/OmniVoice) (zero-shot klonování
hlasu, 646 jazyků, Apache-2.0). Vše běží **lokálně** na této stanici, nic se
neposílá do cloudu.

> Stav: **funkční a ověřené pro češtinu** (viz [Ověření](#ověření) níže).

---

## Tento stroj (detekováno)

| Komponenta | Hodnota |
|---|---|
| OS | Windows 11 Pro (build 26200) |
| CPU | AMD Ryzen 9 9900X (12 jader / 24 vláken) |
| RAM | 128 GB |
| GPU | NVIDIA GeForce RTX 4090, 24 GB VRAM (driver 610.47, CUDA 13.3, compute 8.9) |
| Python | 3.12.11 (spravovaný `uv`, izolovaný v `.venv`) |
| Klíčové balíčky | `torch 2.8.0+cu128`, `omnivoice 0.1.5`, `transformers`, `ffmpeg 7.1` |

Naměřeno při generování: **špičkové VRAM ~2 GB**, rychlost ~5–6× nad realtime
(8 s řeči za ~1,3 s). Na 4090 je tedy obrovská rezerva.

---

## Instalace (už hotová)

Prostředí je připravené. Pokud bys ho potřeboval postavit znovu:

```powershell
uv venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .\.venv\Scripts\python.exe omnivoice soundfile
```

Váhy modelu (~stovky MB) se stáhnou automaticky z Hugging Face při prvním
spuštění do `C:\Users\henri\.cache\huggingface`.

---

## Rychlý start

```powershell
# 1) Jednorázová syntéza textu na řeč (auto hlas)
.\.venv\Scripts\python.exe scripts\say.py "Ahoj světe, dnes je krásný den."

# 2) Z textového souboru
.\.venv\Scripts\python.exe scripts\say.py --file examples\demo_cs.txt --out output\demo.wav

# 3) Klonování hlasu (3–10 s český referenční vzorek)
.\.venv\Scripts\python.exe scripts\say.py "Klonovaný hlas." --ref refs\my_voice.wav --ref-text "přepis referenčního vzorku"

# 4) Dabing z titulků (.srt) — zarovnaná zvuková stopa
.\.venv\Scripts\python.exe scripts\dub_video.py --srt examples\example.srt --ref refs\auto_cs.wav --out-audio output\dub.wav

# 5) Dabing celého videa (nahradí původní zvuk)
.\.venv\Scripts\python.exe scripts\dub_video.py --srt subs.srt --ref refs\voice.wav --video film.mp4 --out-video output\film_cs.mp4
```

---

## Klonování hlasu pro češtinu (důležité)

- Pro **čistou češtinu klonuj z českého referenčního vzorku** (3–10 s, čistý
  zvuk). Když použiješ cizojazyčný hlas, čeština ponese jeho přízvuk.
- Referenční přepis (`--ref-text`) je volitelný — bez něj si ho model přepíše
  sám Whisperem. Předáním přepisu generování zrychlíš a zpřesníš.
- Bez reference dostane **každá věta jiný náhodný hlas** — proto je u dabingu
  reference prakticky povinná. V repu je demo `refs\auto_cs.wav`.

## Český normalizér textu

`src\tts_service\cz_normalize.py` převádí psané tvary na mluvenou češtinu se
správným skloňováním **před** syntézou (model sám nenormalizuje):

| Vstup | Výstup |
|---|---|
| `Mám 3 jablka a 21 hrušek.` | `Mám tři jablka a dvacet jedna hrušek.` |
| `1 234,50 Kč a sleva 15 %` | `tisíc dvě stě třicet čtyři celá pět nula korun a sleva patnáct procent` |
| `21 °C`, `2 hodiny` | `dvacet jeden stupeň Celsia`, `dvě hodiny` |
| `Viz str. 12, kap. 3, atd.` | `viz strana dvanáct, kapitola tři, a tak dále` |

Zapíná se automaticky (vypneš `--no-normalize`). **Limity** (záměrně jednoduché):
neřeší řadové číslovky (`1.` → `první`), data, a pád u méně častých
podstatných jmen aproximuje pravidlem 1 / 2–4 / 5+.

---

## Dabingový workflow

Tento repozitář pokrývá **syntézu a zarovnání** (kroky 4–5). Kompletní dabing:

1. **Přepis** originálu (ASR, např. Whisper) → titulky s časy.
2. **Překlad** do češtiny.
3. **Segmentace** podle časů (`.srt`).
4. **Syntéza** každého segmentu klonovaným hlasem — `dub_video.py`.
5. **Zarovnání + mux** do videa přes ffmpeg — `dub_video.py --video`.

Klíč k synchronizaci: každý segment se vyrenderuje na **přesnou délku slotu**
titulku (parametr `--fit duration`, model zrychlí/zpomalí řeč). Režim
`--fit natural` nechá přirozené tempo a zarovná jen začátek.

---

## Parametry a ladění

| Parametr | Default | Význam |
|---|---|---|
| `--num-step` | 48 | Difuzní kroky. 32 = rychlé, 64 = nejlepší kvalita. |
| `--guidance` | 2.0 | Síla vedení. 3.0–4.0 = věrnější klonování hlasu. |
| `--duration` | – | Vynutí přesnou délku výstupu (sekundy). |
| `--speed` | – | Rychlost řeči (>1 rychleji). |
| `--keep-original` | vyp. | Smíchá původní zvuk pod dabing (ducking). |

Výstup je vždy **24 kHz mono**.

---

## Ověření

Vygenerované české audio bylo přepsáno zpět Whisperem (ASR round-trip):

- `smoke_cs.wav` → „Ahoj, toto je test české syntézy řeči pomocí modelu
  OmniVoice. Mám 3 jablka a celková cena je 1234 korun.“
- `clone_cs.wav` (klonovaný hlas) → „Toto je klonovaný český hlas. Dnes je
  krásný den a mám 42 korun.“ (přesná shoda)

Diakritika i normalizace čísel prošly správně. Spustit znovu:
`.\.venv\Scripts\python.exe scripts\verify_czech.py`

---

## Struktura projektu

```
tts-service/
├─ .venv/                     izolované prostředí (uv, Python 3.12)
├─ src/tts_service/
│  ├─ cz_normalize.py         český normalizér textu (čísla, zkratky, jednotky)
│  ├─ tts.py                  wrapper CzechTTS nad OmniVoice
│  └─ dub.py                  SRT dabing: parsování, zarovnání, mux (ffmpeg)
├─ scripts/
│  ├─ say.py                  CLI: text → WAV
│  ├─ dub_video.py            CLI: SRT (+ video) → dabing
│  ├─ smoke_test.py           rychlý test EN + CZ
│  └─ verify_czech.py         ověření kvality (ASR round-trip + klonování)
├─ examples/                  ukázkový .srt a text
├─ refs/                      referenční hlasy (.wav)
└─ output/                    vygenerované audio/video
```

---

## Právní a etické upozornění

OmniVoice (a tedy i tato služba) **zakazuje neautorizované klonování a
napodobování hlasu**. Dabuj jen **vlastní obsah** nebo hlas, ke kterému máš
souhlas/práva. Klonování hlasu reálné osoby bez souhlasu je proti podmínkám
modelu i právně rizikové (osobnostní práva, hlas jako biometrický údaj).
Licence kódu i modelu: Apache-2.0 (komerční použití povoleno).
