# straigt

Find straight roads and optionally visualise them on an interactive map.

## Usage

```
python find_straight_ways.py pbf/saarland-latest.osm.pbf \
    --min-length 1000 --min-straightness 0.99 --map result.html
```

The `--map` option writes an HTML file with the found ways drawn using
[Folium](https://python-visualization.github.io/folium/). Install the
dependency with `pip install folium` if it is not already available.
