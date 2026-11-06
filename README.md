# Swing-Scanner

Privates Werkzeug, das Aktien nach festen Regeln auf Swingtrade-Setups prüft.
Stufe 1 ist der **Backtest**: Er zeigt, wie gut die Regeln in der Vergangenheit
funktioniert hätten – bevor man den Signalen im Alltag vertraut.

Keine Anlageberatung. Vergangene Ergebnisse sind keine Garantie.

## Was der Backtest prüft

- **Universen:** DAX, MDAX, SDAX (Deutschland) und S&P 500 (USA, als Gegenprobe)
- **Filter:** Index über der 200-Tage-Linie, Aktie im Aufwärtstrend, passende
  Volatilität, genug Umsatz, keine Quartalszahlen in der Haltedauer
- **Setups:** Ausbruch über das 20-Tage-Hoch mit Volumen; Rücksetzer an die
  20-Tage-Linie mit Umkehrsignal
- **Ausstieg:** Stop, Ziel (CRV mindestens 2) oder nach 20 Handelstagen
- **Ehrlicher Test:** Regeln werden bis Ende 2021 entwickelt, 2022 bis heute ist
  der Prüfzeitraum (Out-of-Sample)
- **Zufalls-Benchmark:** Bringt das Signal mehr als ein zufälliger Einstieg im
  selben Umfeld?
- **Portfolio:** 1 % Risiko je Trade, maximal 5 Positionen und 2 je Sektor

Alle Regeln stehen in `config/strategy.yaml` und lassen sich dort ändern.

## Einrichtung (nur über die GitHub-Weboberfläche)

1. **Repository anlegen:** github.com → „New repository“ → Name z. B.
   `swing-scanner`. Öffentlich ist kostenlos und unbegrenzt; privat geht auch
   (2.000 Actions-Minuten pro Monat frei, ein Backtest braucht 5–20 Minuten).
2. **Dateien hochladen:** „Add file“ → „Upload files“ → alle Dateien und Ordner
   aus dem ZIP hineinziehen → „Commit changes“.
   *Achtung:* Ordner mit Punkt am Anfang (`.github`) werden beim Hochziehen oft
   übersprungen. Prüfe danach, ob `.github/workflows/backtest.yml` da ist.
   Falls nicht: „Add file“ → „Create new file“ → als Namen
   `.github/workflows/backtest.yml` eintippen und den Inhalt hineinkopieren.
3. **Schreibrechte für Actions:** Settings → Actions → General → „Workflow
   permissions“ → „Read and write permissions“ → Save.
4. **Seite einschalten:** Settings → Pages → „Deploy from a branch“ → Branch
   `main`, Ordner `/docs` → Save. Die Seite erscheint unter
   `https://<dein-name>.github.io/swing-scanner/`.

## Backtest starten

Actions → „Backtest“ → „Run workflow“ → Markt wählen (`de`, `us` oder
`both`) → „Run workflow“.

Der erste Lauf lädt alle Kurse und dauert länger (DE etwa 5–10 Minuten, US
etwa 15–25 Minuten). Danach liegt der Report unter
`https://<dein-name>.github.io/swing-scanner/backtests/` und zusätzlich als
Download beim Workflow-Lauf.

## Report lesen

- **R** = Vielfaches des Risikos. +1 R heißt: so viel gewonnen, wie beim Stop
  verloren worden wäre.
- **Erwartung je Trade** ist die wichtigste Zahl. Sie sollte im
  Out-of-Sample klar über null liegen, und das 95 %-Intervall sollte null nicht
  einschließen.
- **Zufalls-Benchmark:** Schneiden zufällige Einstiege ähnlich gut ab, kommt
  der Vorteil vom Filter (Trend, Marktumfeld), nicht vom Einstiegssignal.
- **Ampel oben im Report:** grün = messbarer Vorteil, gelb = positiv, aber
  nicht belegt, rot = kein Vorteil.

## Grenzen

- Kursdaten von Yahoo Finance (inoffiziell), Stooq als Ersatz.
- Getestet werden die **heutigen** Indexmitglieder (Survivorship Bias – die
  Ergebnisse sind eher zu gut).
- Termine der Quartalszahlen liefert Yahoo nicht für alle deutschen Werte; der
  Report zeigt, für wie viele sie vorlagen.
- Die SDAX-Liste hat den Stand Dezember 2025 und sollte gelegentlich
  aktualisiert werden (`config/universes/sdax.csv`).

## Lokal ausführen (optional)

```bash
pip install -r requirements.txt
python run_backtest.py --market de                      # echte Daten
python run_backtest.py --market de --source synthetic   # Test ohne Internet
python -m pytest -q
```
