s = """
City: İlanların bulunduğu şehir veya büyük metropol alanı. Örneğin, Toronto, Markham, Oakville gibi çevre banliyölerden ilanları içerebilir.
Price: Kanada doları cinsinden mülkün listelenen fiyatı.
Address: İlan için sokak adresi ve varsa daire numarası.
Number_Beds: İlanda belirtilen yatak odası sayısı.
Number_Baths: İlanda belirtilen banyo sayısı.
Province: Her şehrin bulunduğu il. Not: Ottawa gibi sınır şehirleri, Gatineau gibi sınırdışı şehirlerden ilan içermez.
Population: Şehir nüfusu. simplemaps'ten alınan verilere göre (https://simplemaps.com/data/canada-cities)
Longitude / Latitude: Şehirlerin boylam ve enlem verileri, simplemaps'ten alınmıştır (https://simplemaps.com/data/canada-cities)
Median_Family_Income: Şehir için 2021 Kanada nüfus sayımından alınan ortanca hane geliri.
"""

for i in s.split("\n"):
    print('<span style="color: red;">' + i.split(': ')[0] + ':</span>' + i.replace(i.split(': ')[0] + ':', '', 1) if i != "" else "")

#=====================================================================================================================#
    
import pandas as pd

data = pd.read_csv('HouseListings-Top45Cities-10292023-kaggle.csv', encoding='latin1')
df = pd.DataFrame(data)

provinces = list(df['Province'].value_counts().index)

for i in range(len(provinces)):
    provinces[i] = 'Province_' + provinces[i]

print(provinces)

#=====================================================================================================================#

