"""
Script to generate random maps for multi-agent routing optimization.
Maps follow the format:
- n: number of points
- m: number of agents (2-5)
- tmax: time budget
- First line: start point (x, y, 0)
- Middle lines: reward points (x, y, value 1-10)
- Last line: end point (x, y, 0)
"""

import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt


def find_next_available_filename(maps_dir: Path) -> str:
    """Find the next available filename in the maps directory."""
    existing_files = list(maps_dir.glob("*.txt"))
    existing_numbers = []
    
    for file in existing_files:
        try:
            num = int(file.stem)
            existing_numbers.append(num)
        except ValueError:
            continue  # Skip non-numeric filenames
    
    if not existing_numbers:
        return "1.txt"
    
    next_num = max(existing_numbers) + 1
    return f"{next_num}.txt"


def generate_random_positions(n: int, area_size: Tuple[float, float] = (12, 10), 
                            min_distance: float = 0.5) -> List[Tuple[float, float]]:
    """
    Generate n random positions ensuring minimum distance between points.
    
    Args:
        n: Number of positions to generate
        area_size: (width, height) of the area
        min_distance: Minimum distance between any two points
        
    Returns:
        List of (x, y) tuples
    """
    positions = []
    width, height = area_size
    max_attempts = 1000
    
    for i in range(n):
        attempts = 0
        while attempts < max_attempts:
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            new_pos = (x, y)
            
            # Check minimum distance constraint
            if all(np.sqrt((x - px)**2 + (y - py)**2) >= min_distance 
                  for px, py in positions):
                positions.append(new_pos)
                break
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: Could not place point {i+1} with min_distance constraint.")
            # Place it anyway if we can't satisfy constraint
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            positions.append((x, y))
    
    return positions


def generate_map_data(n: int, m: int, area_size: Tuple[int, int]) -> Tuple[int, int, int, List[Tuple[float, float, int]]]:
    """
    Generate a complete map with n points and m agents.
    
    Args:
        n: Number of points
        m: Number of agents
        area_size: (width, height) of the area
        
    Returns:
        Tuple of (n, m, tmax, points_data)
        where points_data is list of (x, y, value) tuples
    """

    # Generate positions
    positions = generate_random_positions(n, area_size)

    # Generate time budget (roughly proportional to number of points and agents)
    area_diag = int(np.sqrt(area_size[0]**2 + area_size[1]**2))
    depot_dist = np.linalg.norm(np.array(positions[0]) - np.array(positions[-1])).astype(int)
    tmax = random.randint(depot_dist, area_diag)
    
    # Create points data: first and last have value 0, middle have values 1-10
    points_data = []
    
    for i, (x, y) in enumerate(positions):
        if i == 0 or i == n - 1:  # First and last points
            value = 0
        else:  # Middle points
            value = random.randint(1, 10)
        
        points_data.append((x, y, value))
    
    return n, m, tmax, points_data


def write_map_file(filename: Path, n: int, m: int, tmax: int, 
                  points_data: List[Tuple[float, float, int]]) -> None:
    """
    Write map data to file in the required format.
    
    Args:
        filename: Output file path
        n: Number of points
        m: Number of agents
        tmax: Time budget
        points_data: List of (x, y, value) tuples
    """
    with open(filename, 'w') as f:
        f.write(f"n {n}\n")
        f.write(f"m {m}\n")
        f.write(f"tmax {tmax}\n")
        
        for x, y, value in points_data:
            f.write(f"{x:.2f} {y:.2f} {value}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random maps for multi-agent routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--count", "-c", type=int, default=1, 
                       help="Number of maps to generate")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Random seed for reproducibility")
    parser.add_argument("--min_points", type=int, default=10,
                       help="Minimum number of points")
    parser.add_argument("--max_points", type=int, default=30,
                       help="Maximum number of points")
    parser.add_argument("--min_agents", type=int, default=2,
                       help="Minimum number of agents")
    parser.add_argument("--max_agents", type=int, default=5,
                       help="Maximum number of agents")
    parser.add_argument("--area_size", type=float, nargs=2, default=(12, 10),
                       help="Size of the area (width height)")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    maps_dir = Path("maps")
    maps_dir.mkdir(exist_ok=True)
    
    print(f"Generating {args.count} maps in {maps_dir}/")
    
    for i in range(args.count):
        # Generate random parameters
        n = random.randint(args.min_points, args.max_points)
        m = random.randint(args.min_agents, args.max_agents)
        
        # Generate map data
        n, m, tmax, points_data = generate_map_data(n, m, args.area_size)
        
        # Find next available filename
        filename = find_next_available_filename(maps_dir)
        filepath = maps_dir / filename
        
        # Write map file
        write_map_file(filepath, n, m, tmax, points_data)
        
        print(f"Generated map {i+1}/{args.count}: {filename} "
              f"(n={n}, m={m}, tmax={tmax})")
        
    print("Map generation completed!")


if __name__ == "__main__":
    main()