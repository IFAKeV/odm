# OSM Data Mining

In diesem Repo finden sich verschiedene OSM Data Mining Spielereien.

## straight

Find straight roads and optionally visualise them on an interactive map.
Führte nicht zum erwünschten Ergebnis. Was aber nicht an den Scripten liegt oder lag. In meiner Erinnerung war die Straße einfach gerader, als sie es wirklich ist. Dafür weiss ich jetzt wo die geradesten Straßen in NRW sind und könnte sie auch überall sonst auf der Welt finden.

### Usage

```
python find_straight_ways.py pbf/saarland-latest.osm.pbf \
    --min-length 1000 --min-straightness 0.99 --map result.html

# advanced version joining adjacent segments
python find_straight_ways_v02.py pbf/saarland-latest.osm.pbf \
    --min-length 1000 --min-straightness 0.99 --no-secondary

# experimental version using Shapely for geometry merging
python find_straight_ways_v04.py pbf/saarland-latest.osm.pbf \
    --min-length 1000 --min-straightness 0.99 --simplify 0.0001
```

The `--map` option writes an HTML file with the found ways drawn using
[Folium](https://python-visualization.github.io/folium/). Install the
dependency with `pip install folium` if it is not already available.

In `find_straight_ways_v02.py` motorways are ignored and primary or secondary
roads can be excluded using the `--no-primary` and `--no-secondary` options.

`find_straight_ways_v04.py` uses [Shapely](https://shapely.readthedocs.io/) to
merge connected road segments irrespective of name tags before searching for
straight runs. An optional simplification step can remove small kinks from the
geometry. The script remains experimental and may require tuning the
`--simplify` parameter depending on the region.

## Find house number 8 north of unclassified roads

`find_housenumber8_north.py` searches for buildings with `addr:housenumber=8` that
lie north of the nearest `highway=unclassified` segment at a distance between
50–150 m. The result is written to an interactive HTML map with separate layers
for roads and buildings.

For best performance pre-filter the PBF file:

```
osmium tags-filter pbf input.osm.pbf \
    w/highway=unclassified w/building addr:housenumber=8 -o filtered.pbf
python find_housenumber8_north.py filtered.pbf --out houses.html
```

Alternatively let the script run the filtering step:

```
python find_housenumber8_north.py input.osm.pbf --prefilter --out houses.html
```

The script indexes road segments in an R-tree and evaluates building distances
in parallel. Use `--processes` to control the worker count.


Dies war eine wilde Idee, die aber keine sauberen Ergebnisse brachte und daher nicht zum Ziel führte.
Die wirkliche Lösung war Hausnummer 6 und 8 in einer bestimmten Entfernen zueinander zu suchen. Das brachte eine überschaubare Trefferliste und führte mich zum gesuchten Ort. Das aber nicht hier, sondern bei Overpass-Turbo.eu.

## Lonely

Finde die alleinstehendsten Häuser