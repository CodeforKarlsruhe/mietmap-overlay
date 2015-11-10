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

import clusterpolate
import matplotlib.cm
import numpy as np
import requests


# URL of the JSON file exported by the scraper
JSON_URL = 'http://karlsruhe.codefor.de/mieten/mieten.json'

HEATMAP_FILE = 'heatmap.png'
HEATMAP_AREA = ((8.28, 49.08), (8.53, 48.92))
HEATMAP_SIZE = (250, 160)
HEATMAP_COLORMAP = matplotlib.cm.summer
HEATMAP_RADIUS = 0.01


def get_data():
    """
    Load scraped data.

    Returns an array of data points. Each row contains a data point, and
    each data point has a longitude, a lattitude and the average rent per
    square meter.
    """
    return np.array(requests.get(JSON_URL).json())[:, (1, 0, 2)]


def sanitize_data(data, max_rel_dist=6):
    """
    Sanitize data by removing outliers.
    """
    # See https://stackoverflow.com/a/16562028/857390
    values = data[:, 2]
    abs_dist = np.abs(values - np.median(values))
    med_dist = np.median(abs_dist)
    rel_dist = abs_dist / med_dist if med_dist else 0
    return data[rel_dist < max_rel_dist, :]


def create_heatmap(data):
    img = clusterpolate.image(data[:, :2], data[:, 2], size=HEATMAP_SIZE,
                              area=HEATMAP_AREA, radius=HEATMAP_RADIUS,
                              colormap=HEATMAP_COLORMAP)[3]
    img.save(HEATMAP_FILE)


if __name__ == '__main__':
    data = get_data()
    data = sanitize_data(data)
    create_heatmap(data)

