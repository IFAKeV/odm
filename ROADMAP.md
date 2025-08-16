# Roadmap – Finde die längsten geraden Straßenabschnitte in Kartenmaterial der OpenStreetMap

## Ziel
Entwicklung eines Tools zur Identifikation und Visualisierung der längsten geraden Straßenabschnitte in OpenStreetMap-Daten.  
**Wichtige Vorgabe:** Jede Version muss vollständig lauffähig sein und mit einem mitgelieferten Testdatensatz (kleine PBF-Datei, z. B. Bremen) **automatisch getestet** werden können.

---

## Phasen

### Phase 1 – Setup
- [ ] Repository-Struktur erstellen (`py/`, `pbf/`, `docs/`).
- [ ] `requirements.txt` mit allen benötigten Libraries (osmium, shapely, geopandas, folium).
- [X] Kleine PBF-Testdatei (z. B. `bremen-latest.osm.pbf`) ins Repo aufnehmen.
- [ ] Erstes Dummy-Script `find_straightest.py`, das nur die PBF lädt und GeoJSON exportiert.
- [ ] Automatischer Testlauf im Repo: `python3 py/find_straightest.py --pbf pbf/bremen-latest.osm.pbf --out test_out`

### Phase 2 – Segment-Analyse
- [ ] OSM-Ways für relevante `highway=*`-Typen einlesen.
- [ ] Segmentbildung mit Länge und Straightness-Berechnung.
- [ ] Filterung nach Parametern (`--min-len`, `--min-straight`).
- [ ] Automatischer Testlauf: Ausgabe eines GeoJSON mit gefilterten Segmenten.

### Phase 3 – Segment-Verknüpfung
- [ ] Geradheits-basierte Verbindung benachbarter Segmente (Winkel-/Distanz-Toleranz).
- [ ] Berechnung zusammenhängender Runs und deren Gesamtlänge.
- [ ] Automatischer Testlauf: Export Top-N längster Runs als GeoJSON.

### Phase 4 – Visualisierung
- [ ] Generierte Segmente und Runs farblich markieren.
- [ ] Interaktive Karte (Leaflet/Folium) mit Popup-Infos (Länge, Straightness, ID).
- [ ] Automatischer Testlauf: Generierte HTML-Karte prüfen (Datei-Existenz + Inhalt).

### Phase 5 – Validierung & Optimierung
- [ ] Parameter justierbar: `--min-len`, `--min-straight`, `--angle-tol`.
- [ ] Performance-Optimierungen für mittlere PBFs (<100 MB).
- [ ] Regressionstests mit festen Parametern.

---

## Verbindliche Anforderungen an Codex
1. **Kein Pseudocode.** Jede Version muss direkt lauffähig sein.  
2. **Automatisch testbar.** Mitgelieferter PBF-Datensatz (Bremen) muss ohne externe Downloads funktionieren.  
3. **Ausgabe sichtbar.** Jedes Script erzeugt GeoJSON + HTML-Karte.  
4. **Eigenständig getestet.** Codex muss seine Lösung mit den Testdaten **selbst ausführen** und funktionierende Ergebnisse sicherstellen, bevor er sie bereitstellt.  

---

## Teststrategie
- Haupttest: `python3 py/find_straightest.py --pbf pbf/bremen-latest.osm.pbf --out-prefix out`
- Erwartete Artefakte:
  - `out_segments.geojson` (Segmente mit Attributen)
  - `out_map.html` (Interaktive Karte)
- Akzeptanzkriterium: Dateien werden erzeugt und enthalten sichtbare gerade Straßenstücke.
