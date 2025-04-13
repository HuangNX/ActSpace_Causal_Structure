import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import pickle

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Set KDE parameters via command line.")
    
    # Define arguments for KDE parameters
    parser.add_argument('--bandwidth', type=float, default=1, help="Bandwidth for Kernel Density Estimation")
    parser.add_argument('--leaf_size', type=int, default=40, help="Leaf size for Kernel Density Estimation")
    parser.add_argument('--kde_algorithm', type=str, default='kd_tree', choices=['kd_tree', 'ball_tree', 'brute'], 
                        help="Algorithm to use for Kernel Density Estimation")
    parser.add_argument('--metric', type=str, default='euclidean', help="Metric to use for Kernel Density Estimation")
    parser.add_argument('--kernel', type=str, default='gaussian', choices=['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'], 
                        help="Kernel to use for Kernel Density Estimation")
    
    parser.add_argument('--eps', type=float, default=0.5, help="Neighborhood radius parameter for DBSCAN")
    parser.add_argument('--min_samples', type=int, default=5, help="Minimum number of samples required for DBSCAN")

    parser.add_argument('--data_path', type=str, default='../data/stay_data_23_24_timeinfo.csv', help="Path to the input data file")

    return parser.parse_args()

# Calculate period points for each individual
def calculate_period_points(group):
    first_row = group.iloc[0]
    home_time = first_row['home_time']
    work_time = first_row['work_time']
    
    if home_time >= 0 and work_time >= 0:
        t_min = min(home_time, work_time)
        t_max = max(home_time, work_time)
        period_points = {
            'period0_end': t_min - 4,
            'period1_end': t_max,
            'period2_end': t_max + 8
        }
    elif home_time >= 0:
        period_points = {
            'period0_end': home_time - 4,
            'period1_end': home_time,
            'period2_end': home_time + 8
        }
    elif work_time >= 0:
        period_points = {
            'period0_end': work_time - 4,
            'period1_end': work_time,
            'period2_end': work_time + 8
        }
    else:
        period_points = {
            'period0_end': np.nan,
            'period1_end': np.nan, 
            'period2_end': np.nan
        }
    
    group['period'] = 0
    group.loc[(group['t'] >= period_points['period0_end']) & (group['t'] < period_points['period1_end']), 'period'] = 1
    group.loc[(group['t'] >= period_points['period1_end']) & (group['t'] < period_points['period2_end']), 'period'] = 2
    group.loc[group['t'] >= period_points['period2_end'], 'period'] = 3
    
    return group

def vectorized_period_assignment(full_data, relocate_info_filter):
    # Precompute period parameters for each pid
    period_params = relocate_info_filter[['pid', 'home_time', 'work_time']].drop_duplicates()
    
    # Compute time points for the three cases
    mask_both = (period_params['home_time'] >= 0) & (period_params['work_time'] >= 0)
    mask_home = (period_params['home_time'] >= 0) & (period_params['work_time'] < 0)
    mask_work = (period_params['home_time'] < 0) & (period_params['work_time'] >= 0)

    period_params['period0_end'] = np.select(
        [mask_both, mask_home, mask_work],
        [
            np.minimum(period_params['home_time'], period_params['work_time']) - 4,
            period_params['home_time'] - 4,
            period_params['work_time'] - 4
        ],
        default=np.nan
    )
    
    period_params['period1_end'] = np.select(
        [mask_both, mask_home, mask_work],
        [
            np.maximum(period_params['home_time'], period_params['work_time']),
            period_params['home_time'],
            period_params['work_time']
        ],
        default=np.nan
    )
    
    period_params['period2_end'] = np.select(
        [mask_both, mask_home, mask_work],
        [
            np.maximum(period_params['home_time'], period_params['work_time']) + 8,
            period_params['home_time'] + 8,
            period_params['work_time'] + 8
        ],
        default=np.nan
    )

    # Merge into full_data
    full_data = full_data.merge(
        period_params[['pid', 'period0_end', 'period1_end', 'period2_end']], 
        on='pid', 
        how='left'
    )

    # Period assignment using vectorized conditions
    conditions = [
        full_data['t'] < full_data['period0_end'], 
        (full_data['t'] >= full_data['period0_end']) & (full_data['t'] < full_data['period1_end']),
        (full_data['t'] >= full_data['period1_end']) & (full_data['t'] < full_data['period2_end']),
        full_data['t'] >= full_data['period2_end']
    ]
    
    full_data['period'] = np.select(conditions, [0, 1, 2, 3], default=0)
    
    # Clean up intermediate columns
    full_data.drop(columns=['period0_end', 'period1_end', 'period2_end'], inplace=True)
    
    return full_data


def compute_activity_area_optimized(ss_grid, density, dbscan_params, threshold_rate=0.5, precomputed_area=None):
    # Optimized coverage area computation function
    if len(density) != len(ss_grid):
        raise ValueError("Density array length must match ss_grid row count")
    if not (0 <= threshold_rate <= 1):
        raise ValueError("Threshold rate must be between 0 and 1")

    area = precomputed_area if precomputed_area is not None else ss_grid.geometry.area.values

    max_density = np.nanmax(density)  
    threshold = threshold_rate * max_density
    mask = density >= threshold
    
    valid_density = density[mask]
    valid_area = area[mask]
    valid_grid = ss_grid[mask].copy()
    
    max_den, min_den = np.max(valid_density), np.min(valid_density)
    # calculate density-weighted area
    weighted_sum = np.dot((valid_density-min_den)/(max_den-min_den), valid_area)

    # #####DBScan CLustering##########
    coords = np.array([valid_grid.chess_x, valid_grid.chess_y]).T
    dbscan = DBSCAN(**dbscan_params)
    valid_grid['cluster'] = dbscan.fit_predict(coords)

    # calculate clustering counts
    cluster_count = len(np.unique(valid_grid['cluster'])) - (1 if -1 in valid_grid['cluster'].values else 0)
    
    return weighted_sum / (1000**2), cluster_count  # Convert to square kilometers



def process_single_pid(pid, pid_data, kde_params, dbscan_params):
    pid_areas = []
    cluster_counts = []
    for period in range(4):
        period_data = pid_data[pid_data['period'] == period]
        if len(period_data) < 2:
            pid_areas.append(0.0)
            continue
            
        data_scaled = period_data[['x_scaled', 'y_scaled']].values
        weights = period_data['frequency'].values
        
        kde = KernelDensity(**kde_params)
        kde.fit(data_scaled, sample_weight=weights)
        
        log_density = kde.score_samples(eval_points)
        density = np.exp(log_density)
        
        coverage_area, cluster_count = compute_activity_area_optimized(
            ss_grid_scaled, density, dbscan_params,
            threshold_rate=0.3,
            precomputed_area=precomputed_area
        )
        pid_areas.append(coverage_area)
        cluster_counts.append(cluster_count)

    return (pid, pid_areas, cluster_counts)

def parallel_processing(data, kde_params, dbscan_params, n_jobs=-1, chunk_size=100):
    groups = [(pid, group) for pid, group in data.groupby('pid')]
    tasks = (delayed(process_single_pid)(pid, group, kde_params, dbscan_params) for pid, group in groups)
    
    results = Parallel(n_jobs=n_jobs, batch_size=chunk_size)(
        tqdm(tasks, total=len(groups), desc='Processing')
    )
    
    return {pid: (areas, cluster_count) for pid, areas, cluster_count in results}


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Set KDE and DBSCAN parameters from command-line arguments
    kde_params = {
        'bandwidth': args.bandwidth,
        'algorithm': args.kde_algorithm,
        'metric': args.metric,
        'leaf_size': args.leaf_size,
        'kernel': args.kernel
    }

    dbscan_params = {
    'eps': 1,
    'min_samples': 3
    }

    # Load grid data
    grid_path = '../../autodl-fs/ss_city_grid_by_cover.csv'
    ss_grid_data = pd.read_csv(grid_path)

    # Convert ss_grid_data into GeoDataFrame
    ss_grid_data['wkt'] = ss_grid_data['wkt'].apply(wkt.loads)
    ss_grid_data = gpd.GeoDataFrame(ss_grid_data, geometry=ss_grid_data.wkt, crs='EPSG:4326')

    # Load stay data
    path = args.data_path
    full_data_period = pd.read_csv(path)

    spatial_bandwidth = 1000

    # Convert full data to GeoDataFrame
    full_data_gpd = gpd.GeoDataFrame(full_data_period, geometry=gpd.points_from_xy(full_data_period.lon, full_data_period.lat), 
                                    crs='EPSG:4326')

    # Reproject to EPSG:3857
    full_data_gpd = full_data_gpd.to_crs(epsg=3857)
    full_data_gpd['x'] = full_data_gpd.geometry.x
    full_data_gpd['y'] = full_data_gpd.geometry.y

    # Precompute scaled coordinates
    full_data_gpd['x_scaled'] = full_data_gpd['x'] / spatial_bandwidth
    full_data_gpd['y_scaled'] = full_data_gpd['y'] / spatial_bandwidth

    # Precompute grid evaluation points
    ss_grid_scaled = ss_grid_data.copy()
    ss_grid_scaled = ss_grid_scaled.to_crs(epsg=3857)

    ss_grid_scaled['centroid'] = ss_grid_scaled.geometry.centroid
    ss_grid_scaled['x'] = ss_grid_scaled['centroid'].x
    ss_grid_scaled['y'] = ss_grid_scaled['centroid'].y
    ss_grid_scaled['x'] /= spatial_bandwidth
    ss_grid_scaled['y'] /= spatial_bandwidth
    eval_points = ss_grid_scaled[['x', 'y']].values
    ss_grid_scaled['precomputed_area'] = ss_grid_scaled.geometry.area
    precomputed_area = ss_grid_scaled.geometry.area.values

    activity_space_areas_dict = {}

    # Precompute all unique pid-period combinations
    all_pids = full_data_gpd['pid'].unique()
    total_tasks = len(all_pids)

    activity_space_areas_dict = parallel_processing(full_data_gpd, kde_params, dbscan_params)

    # Save results
    with open('activity_space_areas_dict.pkl', 'wb') as f:
        pickle.dump(activity_space_areas_dict, f)