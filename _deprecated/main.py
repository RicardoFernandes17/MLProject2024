import pandas as pd
import matplotlib.pyplot as plt
from helper import get_dataset_with_copy,get_columns_with_nulls, fill_null_values, get_date_values_from_timestamp, get_jaguar_movement_features,create_time_window_features,calculate_movement_features,classify_movement_state,process_dataset

import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeatures
from jaguar_feature_scaling import get_speed_feature_plot,get_distance_distribution_plot,get_movement_patterns_plot,get_direction_distribution_plot, get_temporal_analysis_plot

# This will be set to see most of the infomation of any print that i make
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000); 

jaguar_data_original, jaguar_data = get_dataset_with_copy('data/raw/jaguar_movement_data.csv')
jaguar_info_original, jaguar_info = get_dataset_with_copy('data/raw/jaguar_additional_information.csv')

jaguar_data_nulls = get_columns_with_nulls(jaguar_data_original)
jaguar_info_nulls = get_columns_with_nulls(jaguar_info_original)

print("Jaguar Data nulls",get_columns_with_nulls(jaguar_data_original))

print("Jaguar info nulls",get_columns_with_nulls(jaguar_info_original))

if len(jaguar_data_nulls) > 0:
    jaguar_data = fill_null_values(jaguar_data, jaguar_data_nulls)

if len(jaguar_info_nulls) > 0:
    jaguar_info = fill_null_values(jaguar_info, jaguar_info_nulls)
    
# Validate the results
#print(get_columns_with_nulls(jaguar_data))
#print(get_columns_with_nulls(jaguar_info))

# Pre process dataset
jaguar_data = process_dataset(
    jaguar_data, 
    {'location.long': 'longitude', 'location.lat': 'latitude', 'individual.local.identifier (ID)': 'individual_id' }, 
    ['Event_ID', 'individual.taxon.canonical.name','tag.local.identifier', 'study.name', 'country']
)

jaguar_info = process_dataset(
    jaguar_info, 
    {'ID':'individual_id','Sex': 'sex', 'Estimated Age': 'age', 'Weight': 'weight' }, 
    ['Collar Type', 'Collar Brand','Planned Schedule', 'Project Leader', 'Contact']
)

# Convert timestamp to datetime.
jaguar_data['timestamp'] = pd.to_datetime(jaguar_data['timestamp'], errors='coerce')

#print(jaguar_data.columns.to_list())
#print(jaguar_info.columns.to_list())
#print(jaguar_data.dtypes)
#print(jaguar_info.dtypes)
#print(jaguar_data.head())
#print(jaguar_info.head())

jaguar_datanew = jaguar_data.merge(jaguar_info, on='individual_id', how='left')

jaguar_datanew = get_date_values_from_timestamp(jaguar_datanew)

jaguar_datanew = get_jaguar_movement_features(jaguar_datanew)

#print(jaguar_datanew[['individual_id', 'timestamp', 'latitude', 'longitude', 'distance', 'speed', 'direction']].head(10))

#print(jaguar_datanew['distance'].describe())
#print("-----")
#print(jaguar_datanew['speed'].describe())
#print("-----")
#print(jaguar_datanew['direction'].describe())
#print(jaguar_datanew.columns.to_list())


####################
## Turn into helper
####################
speed_plt = get_speed_feature_plot(jaguar_datanew)
speed_plt.show()
distance_plt = get_distance_distribution_plot(jaguar_datanew)
speed_plt.show()
movement_plt = get_movement_patterns_plot(jaguar_datanew)
speed_plt.show()
direction_plt = get_direction_distribution_plot(jaguar_datanew)
speed_plt.show()
travaled_distance_plt = get_temporal_analysis_plot(jaguar_datanew)
travaled_distance_plt.show()

fig, ax = plt.subplots(figsize=(40, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# Set the extent to the whole world (-180 to 180 longitude, -90 to 90 latitude)
# ax.set_global()

# Add map features
ax.add_feature(cfeatures.LAND, edgecolor='black')
ax.add_feature(cfeatures.OCEAN)
ax.add_feature(cfeatures.COASTLINE)
ax.add_feature(cfeatures.BORDERS, linestyle=':')

# Plot each jaguar's movement

for jaguar_id in jaguar_datanew['individual_id'].unique():
    jaguar_subset = jaguar_datanew[jaguar_datanew['individual_id'] == jaguar_id]
    ax.plot(jaguar_subset['longitude'], jaguar_subset['latitude'], 
             label=f'Jaguar {jaguar_id}', alpha=1)

# Labels and legend
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Jaguar Movement Paths on World Map")
ax.legend()
plt.show()

print("\nMovement Statistics by Time of Day:")
print(jaguar_datanew.groupby('time_of_day')[['speed', 'distance']].agg(['mean', 'std']).round(3))

print("\nMovement Statistics by Individual:")
print(jaguar_datanew.groupby('individual_id')[['speed', 'distance']].agg(['mean', 'std']).round(3))

window_features = []
for jaguar_id in jaguar_datanew['individual_id'].unique():
    # Get data for current jaguar
    jaguar_data = jaguar_datanew[jaguar_datanew['individual_id'] == jaguar_id].copy()
    
    # Calculate window features without setting index beforehand
    window_stats = create_time_window_features(jaguar_data)
    window_stats['individual_id'] = jaguar_id
    window_features.append(window_stats)
    
window_features_df = pd.concat(window_features).reset_index()

window_features_df = calculate_movement_features(window_features_df)

# 3. Add temporal context features
window_features_df['hour'] = window_features_df['timestamp'].dt.hour
window_features_df['is_night'] = (window_features_df['hour'] >= 18) | (window_features_df['hour'] <= 6)
window_features_df['is_peak_activity'] = window_features_df['hour'].isin([5,6,7,17,18,19])

window_features_df['movement_state'] = window_features_df.apply(classify_movement_state, axis=1)

# Let's visualize our new features
plt.figure(figsize=(15, 30))

# Plot 1: Movement States Distribution
plt.subplot(4, 1, 1)
sns.countplot(data=window_features_df, x='movement_state')
plt.title('Distribution of Movement States')
plt.xticks(rotation=45)

# Plot 2: Speed vs Area Covered
plt.subplot(4, 1, 2)
sns.scatterplot(data=window_features_df, x='speed_mean', y='area_covered', 
                hue='movement_state', alpha=0.6)
plt.title('Speed vs Area Covered by Movement State')

# Plot 3: Movement Patterns by Time of Day
plt.subplot(4, 1, 3)
sns.boxplot(data=window_features_df, x='hour', y='movement_intensity')
plt.title('Movement Intensity by Hour')
plt.xticks(rotation=45)

# Plot 4: Path Efficiency Distribution
plt.subplot(4, 1, 4)
sns.boxplot(data=window_features_df, x='movement_state', y='path_efficiency')
plt.title('Path Efficiency by Movement State')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Print summary statistics of our engineered features
print("\nSummary Statistics of Engineered Features:")
print(window_features_df.describe().round(3))

print("\nMovement State Distribution:")
print(window_features_df['movement_state'].value_counts(normalize=True).round(3))