#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

# Copyright (c) 2015 Code for Karlsruhe (http://codefor.de/karlsruhe)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Visualization overlay for renting costs in Karlsruhe.
"""

from __future__ import division, unicode_literals

import codecs
import json

import clusterpolate
import matplotlib.cm
import numpy as np


# Filename for JSON data
JSON_FILE = 'mieten.json'

HEATMAP_FILE = 'overlay.png'
HEATMAP_AREA = ((8.28, 49.08), (8.53, 48.92))
HEATMAP_SIZE = (200, 128)
HEATMAP_COLORMAP = matplotlib.cm.summer
HEATMAP_RADIUS = 0.00015

# Number of entries in the exported colormap
COLORMAP_EXPORT_ENTRIES = 20

# Filename of the exported colormap
COLORMAP_EXPORT_FILE = 'colormap.json'


def load_data(filename):
    """
    Load scraped data.

    ``filename`` is the name of the JSON file to load.

    Returns a 2-column array of data points (longitude and latitude) and
    a 1-column array with the corresponding average rent per square
    meter.
    """
    with codecs.open(filename, 'r', encoding='utf8') as f:
        data = np.array(json.load(f))
    return data[:, (1, 0)], data[:, 2]


def sanitize_data(points, values, max_rel_dist=6):
    """
    Sanitize data by removing outliers.
    """
    # See https://stackoverflow.com/a/16562028/857390
    abs_dist = np.abs(values - np.median(values))
    med_dist = np.median(abs_dist)
    rel_dist = abs_dist / med_dist if med_dist else 0
    keep = rel_dist < max_rel_dist
    return points[keep, :], values[keep]


def create_heatmap(points, values, area):
    """
    Create clusterpolated heatmap.

    Takes a 2-column matrix with point coordinates, a 1-column matrix
    with associated values, and a 2-tuple of 2-tuples describing the
    target area.

    Returns a ``PIL.Image.Image`` instance.
    """
    return clusterpolate.image(
        points, values, size=HEATMAP_SIZE, area=area,
        radius=HEATMAP_RADIUS, colormap=HEATMAP_COLORMAP)[3]


def lonlat_to_world(points):
    """
    Convert longitude and latitude in degrees to world coordinates.

    World coordinates are based on the Web Mercator projection.

    Takes a 2-column matrix with longitude and latitude in degrees and
    returns a 2-column matrix with world coordinates.
    """
    w = np.pi * points / 180  # Convert to radians
    w[:, 0] += np.pi
    w[:, 1] = np.pi - np.log(np.tan(np.pi / 4 + w[:, 1] / 2))
    return w


def world_to_lonlat(points):
    """
    Convert world coordinates to longitude and latitude in degrees.

    World coordinates are based on the Web Mercator projection.

    Takes a 2-column matrix with world coordinates and returns a
    2-column matrix with longitude and latitude in degrees.
    """
    w = np.zeros(points.shape)
    w[:, 0] = points[:, 0] - np.pi
    w[:, 1] = 2 * (np.arctan(np.exp(-(points[:, 1] - np.pi))) - np.pi / 4)
    return w * 180 / np.pi  # Convert to degrees


def export_colormap(cm, filename, entries=20):
    """
    Export a matplotlib colormap to a JSON file.

    The colormap is exported as a list of ``entries`` RGBA tuples for
    linearly spaced indices between 0 and 1 (inclusive).
    """
    with codecs.open(filename, 'w', encoding='utf8') as f:
        json.dump(cm(np.linspace(0, 1, entries)).tolist(), f)


if __name__ == '__main__':
    points, values = load_data(JSON_FILE)
    points, values = sanitize_data(points, values)

    w_points = lonlat_to_world(points)
    w_area = lonlat_to_world(np.array(HEATMAP_AREA))

    img = create_heatmap(w_points, values, w_area)
    img.save(HEATMAP_FILE)

    export_colormap(HEATMAP_COLORMAP, COLORMAP_EXPORT_FILE,
                    COLORMAP_EXPORT_ENTRIES)

