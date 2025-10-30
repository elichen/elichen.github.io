#!/usr/bin/env python3
"""
Generate racing track with constant width using Shapely library.
Usage: source venv/bin/activate && python generate_track.py
"""

from shapely.geometry import Polygon

def create_track_with_shapely(outer_points, track_width=120):
    """Create inner track boundary using Shapely's buffer operation."""
    # Create outer polygon
    outer_polygon = Polygon(outer_points)

    # Create inner polygon by negative buffer (inward offset)
    inner_polygon = outer_polygon.buffer(
        -track_width,
        join_style=2,  # Mitre join for sharp corners
        mitre_limit=2.0,
        resolution=16
    )

    if inner_polygon.is_empty:
        print(f"Error: Track width {track_width}px is too large")
        return None

    # Extract and round coordinates
    coords = list(inner_polygon.exterior.coords)[:-1]
    return [[round(x), round(y)] for x, y in coords]

# Current outer points from track.js (bean shape with upward arch)
outer_points = [
    [-400, -250],  # Top left
    [400, -250],   # Top right
    [500, -150],   # Top right corner
    [500, 50],     # Right side upper
    [480, 150],    # Right side lower
    [400, 230],    # Bottom right curve
    [250, 260],    # Bottom right approaching arch
    [100, 250],    # Bottom right side of arch
    [0, 120],      # Bottom center (deep upward arch)
    [-100, 250],   # Bottom left side of arch (symmetric)
    [-250, 260],   # Bottom left approaching arch (symmetric)
    [-400, 230],   # Bottom left curve (symmetric)
    [-480, 150],   # Left side lower (symmetric)
    [-500, 50],    # Left side upper (symmetric)
    [-500, -150]   # Top left corner
]

# Generate inner boundary
inner_points = create_track_with_shapely(outer_points, track_width=120)

if inner_points:
    print("// Inner points generated with Shapely (constant 120px width)")
    print("this.innerPoints = [")
    for i, point in enumerate(inner_points):
        print(f"    [{point[0]}, {point[1]}],")
    print("]")
else:
    print("Failed to generate inner track boundary")