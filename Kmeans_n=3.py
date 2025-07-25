# === 1. Import Dependencies ===
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import seaborn as sns
import segyio

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from shutil import copyfile

warnings.simplefilter('ignore')
sns.set(style='ticks')

# === 2. Load Seismic Data from .sgy Files ===
def load_seismic_cube(name):
    return segyio.tools.cube(f"{name}.sgy")

attributes = {
    "Inst_Frequency_data": load_seismic_cube('instantaneous_frequency_B-31a-93-TX'),
    "Sweetness_data": load_seismic_cube('sweetness_B-31a-93-TX'),
    "Energy_Ratio_Similarity_data": load_seismic_cube('energy_ratio_similarity_B-31a-93-TX'),
    "GLCM_Energy_data": load_seismic_cube('glcm_energy_B-31a-93-TX'),
}

# Check consistency of shapes
shapes = [data.shape for data in attributes.values()]
for name, shape in zip(attributes.keys(), shapes):
    print(f"{name} shape: {shape}")

if all(shape == shapes[0] for shape in shapes):
    print("All cubes have the same shape.")
else:
    raise ValueError("Mismatch in cube shapes.")

# === 3. Visualize Sample Inlines and Time Slices ===
def plot_section(data, index, title, cmap, vmin=None, vmax=None, slice_axis=0):
    section = np.take(data, index, axis=slice_axis).T
    plt.imshow(section, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.ylabel("Time" if slice_axis == 0 else "Crosslines")
    if slice_axis != 0:
        plt.xlabel("Inlines")
    plt.show()

plot_section(attributes["Inst_Frequency_data"], 150, "Inst_Frequency Inline 150", 'brg', 5, 50, slice_axis=0)
plot_section(attributes["Inst_Frequency_data"], 12, "Inst_Frequency Time Slice at 650 ms", 'brg', 5, 50, slice_axis=2)

plot_section(attributes["Sweetness_data"], 150, "Sweetness Inline 150", 'rainbow', 0, 2500, slice_axis=0)
plot_section(attributes["Sweetness_data"], 12, "Sweetness Time Slice at 650 ms", 'rainbow', 0, 2500, slice_axis=2)

plot_section(attributes["Energy_Ratio_Similarity_data"], 150, "Energy Ratio Similarity Inline 150", 'Greys', 0.95, 1, slice_axis=0)
plot_section(attributes["Energy_Ratio_Similarity_data"], 12, "Energy Ratio Similarity Time Slice at 650 ms", 'Greys', 0.95, 1, slice_axis=2)

plot_section(attributes["GLCM_Energy_data"], 150, "GLCM Energy Inline 150", 'Greys', 0, 0.5, slice_axis=0)
plot_section(attributes["GLCM_Energy_data"], 12, "GLCM Energy Time Slice at 650 ms", 'Greys', 0, 0.5, slice_axis=2)

# === 4. Reshape Cubes into 1D Arrays and Combine ===
reshaped = {k: np.reshape(v, (-1, 1)) for k, v in attributes.items()}
combined_np = np.concatenate(list(reshaped.values()), axis=1)

print("Combined data shape (before decimation):", combined_np.shape)

# === 5. Create DataFrame from Combined Data ===
combined_df = pd.DataFrame(combined_np, columns=reshaped.keys())

# === 6. Subsample for Visualization ===
subset_df = combined_df.iloc[::5000]

# === 7. Pairplot of Subset ===
sns.pairplot(subset_df)
plt.show()

# === 8. Histograms and Boxplots ===
combined_df.hist(bins=50, figsize=(12, 8))
plt.show()

for col in combined_df.columns:
    sns.boxplot(y=combined_df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# === 9. KMeans Clustering (k=3) ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_df)

kmeans = KMeans(n_clusters=3, init='k-means++')
cluster_labels = kmeans.fit_predict(scaled_data)

combined_df['Seismic_Facies_km3'] = cluster_labels

# Subset for cluster visualization
subset_df = combined_df.iloc[::5000]

# === 10. Pairplot with Cluster Labels ===
sns.pairplot(subset_df, vars=list(reshaped.keys()), hue='Seismic_Facies_km3', palette="muted")
plt.show()

# === 11. Cluster Statistics and Boxplots ===
print(combined_df["Seismic_Facies_km3"].value_counts())

combined_df.hist(column=['Seismic_Facies_km3'], bins=3, figsize=(12, 8))
plt.show()

for col in reshaped.keys():
    sns.boxplot(x=combined_df['Seismic_Facies_km3'], y=combined_df[col])
    plt.title(f'{col} by Seismic Facies')
    plt.show()

# === 12. Save Cluster Labels Back to SEG-Y Format ===
original_segy = 'instantaneous_frequency_B-31a-93-TX.sgy'
shape = attributes["Inst_Frequency_data"].shape
volume = np.zeros(shape, dtype='float32')
volume.ravel()[:] = cluster_labels.astype('float32')

output_file = 'Kmeans_k3_volume.sgy'
copyfile(original_segy, output_file)

with segyio.open(output_file, "r+") as dst:
    ilines = dst.ilines
    for j, iline in enumerate(ilines):
        dst.iline[iline] = volume[j, :, :]

def plot_section(data, index, title, slice_axis=0, n_classes=None, class_colors=None):
    # 1. Inferir número de clases si no se pasa como parámetro
    # 1. Infer number of clusters if not specified as parameter
    if n_classes is None:
        n_classes = int(np.max(data)) + 1
    # 2. Generar colores si no se especifican
    # 2. Generate colors if not specified 
    if class_colors is None:
        cmap_base = cm.get_cmap('tab10') if n_classes <= 10 else cm.get_cmap('tab20')
        class_colors = [cmap_base(i) for i in range(n_classes)]
    cmap = ListedColormap(class_colors)
    # 3. Crear límites de clases: por ejemplo para 3 clases → [-0.5, 0.5, 1.5, 2.5]
    # 3. Create clusters limits: for example for 3 clusters → [-0.5, 0.5, 1.5, 2.5]
    bounds = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    # 4. Seleccionar sección
    # 4. select section
    section = np.take(data, index, axis=slice_axis).T
    # 5. Graficar
    # 5. Plot
    im = plt.imshow(section, cmap=cmap, norm=norm, aspect='auto')
    plt.title(title)   
    # 6. Barra de colores con ticks enteros
    # 6. Color bar with integer ticks
    cbar = plt.colorbar(im, ticks=list(range(n_classes)))
    cbar.ax.set_yticklabels([str(i) for i in range(n_classes)])
    # 7. Etiquetas de ejes
    # 7. Axis labels
    plt.ylabel("Time" if slice_axis == 0 else "Crosslines")
    if slice_axis != 0:
        plt.xlabel("Inlines")
    # 8. Mostrar
    # 8. Display
    plt.show()  

plot_section(volume, 250, "KMeans k=3 Inline 250", slice_axis=0)
plot_section(volume, 12, "KMeans k=3 Time Slice at 650 ms", slice_axis=2)