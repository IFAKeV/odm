# straigt

Find straight roads and optionally visualise them on an interactive map.

## Usage

```
python find_straight_ways.py pbf/saarland-latest.osm.pbf \
    --min-length 1000 --min-straightness 0.99 --map result.html

# advanced version joining adjacent segments
python find_straight_ways_v02.py pbf/saarland-latest.osm.pbf \
    --min-length 1000 --min-straightness 0.99 --no-secondary
```

The `--map` option writes an HTML file with the found ways drawn using
[Folium](https://python-visualization.github.io/folium/). Install the
dependency with `pip install folium` if it is not already available.

In `find_straight_ways_v02.py` motorways are ignored and primary or secondary
roads can be excluded using the `--no-primary` and `--no-secondary` options.
