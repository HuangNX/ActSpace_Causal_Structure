import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed
from tqdm import tqdm
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


def compute_activity_area_optimized(ss_grid, density, threshold_rate=0.5, precomputed_area=None):
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
    weighted_sum = np.dot(valid_density, valid_area)
    
    return weighted_sum / (1000**2)  # Convert to square kilometers



def process_single_pid(pid, pid_data, kde_params):
    pid_areas = []
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
        
        coverage_area = compute_activity_area_optimized(
            ss_grid_scaled, density, 
            precomputed_area=precomputed_area
        )
        pid_areas.append(coverage_area)
    return (pid, pid_areas)

def parallel_processing(data, kde_params, n_jobs=-1, chunk_size=100):
    groups = [(pid, group) for pid, group in data.groupby('pid')]
    tasks = (delayed(process_single_pid)(pid, group, kde_params) for pid, group in groups)
    
    results = Parallel(n_jobs=n_jobs, batch_size=chunk_size)(
        tqdm(tasks, total=len(groups), desc='Processing')
    )
    
    return {pid: areas for pid, areas in results}



if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Set KDE parameters from command-line arguments
    kde_params = {
        'bandwidth': args.bandwidth,
        'algorithm': args.kde_algorithm,
        'metric': args.metric,
        'leaf_size': args.leaf_size,
        'kernel': args.kernel
    }

    # Load full data
    full_data = pd.read_csv('../data/stay_data_23_24.csv')
    complete_time = full_data[['year', 'week_number']].drop_duplicates().sort_values(by=['year', 'week_number']).reset_index(drop=True)
    complete_time['t'] = np.arange(len(complete_time))
    full_data = full_data.merge(complete_time, on=['year', 'week_number'], how='left')

    outcome = pd.read_csv('../../autodl-fs/slow_dist_data.csv')
    relocate_info = outcome[['pid', 'home_year', 'home_week', 'home_time', 'work_year', 'work_week', 'work_time']].drop_duplicates()

    # Precompute max time for each individual
    max_t_dict = full_data.groupby('pid')['t'].max().to_dict()

    # Preprocess relocate_info by adding max_t
    relocate_info['max_t'] = relocate_info['pid'].map(max_t_dict)

    # 1. Only home relocation samples
    mask_only_home = (relocate_info['home_time'] >= 0) & (relocate_info['work_time'] < 0)
    mask_home_time = (relocate_info['home_time'] >= 9) & (relocate_info['max_t'] - relocate_info['home_time'] >= 20)
    only_home = relocate_info[mask_only_home & mask_home_time].copy()

    # 2. Only work relocation samples
    mask_only_work = (relocate_info['work_time'] >= 0) & (relocate_info['home_time'] < 0)
    mask_work_time = (relocate_info['work_time'] >= 9) & (relocate_info['max_t'] - relocate_info['work_time'] >= 20)
    only_work = relocate_info[mask_only_work & mask_work_time].copy()

    # 3. Both work and home relocation samples with a gap <= 8 weeks
    mask_both = (relocate_info['home_time'] >= 0) & (relocate_info['work_time'] >= 0)
    both_change = relocate_info[mask_both].copy()
    both_change['time_gap'] = (both_change['home_time'] - both_change['work_time']).abs()

    # Filter samples based on time conditions
    mask_both_time = (
        (both_change['home_time'] >= 9) & 
        (both_change['max_t'] - both_change['home_time'] >= 20) &
        (both_change['work_time'] >= 9) & 
        (both_change['max_t'] - both_change['work_time'] >= 20) &
        (both_change['time_gap'] <= 8)
    )
    both_change = both_change[mask_both_time]

    # Combine the three categories
    relocate_info_filter = pd.concat([only_home[['pid', 'home_time', 'work_time', 'max_t']],
                                    only_work[['pid', 'home_time', 'work_time', 'max_t']],
                                    both_change[['pid', 'home_time', 'work_time', 'max_t']]], ignore_index=True)

    # Filter full_data based on relocate_info_filter pid
    full_data = full_data[full_data['pid'].isin(relocate_info_filter['pid'])]
    print(full_data['pid'].nunique())

    # Merge home and work time columns from relocate_info_filter into full_data
    full_data = pd.merge(full_data, relocate_info_filter[['pid', 'home_time', 'work_time']], on='pid', how='left')

    # Load grid data
    grid_path = '../../autodl-fs/ss_city_grid_by_cover.csv'
    ss_grid_data = pd.read_csv(grid_path)

    # Convert ss_grid_data into GeoDataFrame
    ss_grid_data['wkt'] = ss_grid_data['wkt'].apply(wkt.loads)
    ss_grid_data = gpd.GeoDataFrame(ss_grid_data, geometry=ss_grid_data.wkt, crs='EPSG:4326')

    # Add period column
    full_data_period = vectorized_period_assignment(full_data, relocate_info_filter)

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

    activity_space_areas_dict = parallel_processing(full_data_gpd, kde_params)

    # Save results
    with open('activity_space_areas_dict.pkl', 'wb') as f:
        pickle.dump(activity_space_areas_dict, f)