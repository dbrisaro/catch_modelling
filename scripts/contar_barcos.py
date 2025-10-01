import pandas as pd

df = pd.read_csv('../Trayectoria_2024_I.csv', sep=';')
barcos_unicos = df['Cod_Barco'].nunique()
print(f'Número total de barcos únicos: {barcos_unicos}')

print('\nLista de barcos:')
lista_barcos = df['Cod_Barco'].unique()
for barco in sorted(lista_barcos):
    registros = len(df[df['Cod_Barco'] == barco])
    print(f'{barco}: {registros} registros')

