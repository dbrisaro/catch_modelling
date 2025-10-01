import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('../Trayectoria_2024_I.csv', sep=';')

primeros_5_barcos = sorted(df['Cod_Barco'].unique())[:5]

plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.gridlines(draw_labels=True)

df_5_barcos = df[df['Cod_Barco'].isin(primeros_5_barcos)]

min_lon = df_5_barcos['Lon'].min()
max_lon = df_5_barcos['Lon'].max()
min_lat = df_5_barcos['Lat'].min()
max_lat = df_5_barcos['Lat'].max()

margin_lon = (max_lon - min_lon) * .25
margin_lat = (max_lat - min_lat) * .25

ax.set_extent([
    min_lon - margin_lon,
    max_lon + margin_lon,
    min_lat - margin_lat,
    max_lat + margin_lat
])

colores = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFA500']

for barco, color in zip(primeros_5_barcos, colores):
    datos_barco = df[df['Cod_Barco'] == barco]
    ax.plot(datos_barco['Lon'], datos_barco['Lat'], 
            label=barco, alpha=0.2, linewidth=1, color=color,
            marker='.', markersize=1,
            transform=ccrs.PlateCarree())

plt.title('Trayectorias de los primeros 5 barcos')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('../figures/trayectorias_5_barcos.png', dpi=300, bbox_inches='tight')

print("Barcos visualizados:")
for barco in primeros_5_barcos:
    registros = len(df[df['Cod_Barco'] == barco])
    print(f"{barco}: {registros} registros")