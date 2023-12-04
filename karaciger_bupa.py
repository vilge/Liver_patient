# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



#ADIM 1: VERİ OKUMA VE ANLAMA
#klasörü güncellemeyi unutma!!!
df = pd.read_csv(r"C:\Users\Qua Lilith\Desktop\Liver_patient\bupa.data")
#print(df.head())

#Hasta ve Hasta değil arasındaki ayrımı görmek için DATASET değerlerini PLOTladım
sizes = df['DATASET'].value_counts(sort = 1)
plt.pie(sizes, shadow=True, autopct='%1.1f%%')



#ADIM 2: EKSİK DEĞERLERİN BIRAKILMASI

df = df.dropna()  #En az bir null değeri olan tüm satırları bırakır.



#ADIM 3: VERİLERİN HAZIRLANMASI

#Y, bağımlı değişkenli veridir, bu DATASET sütunudur
Y = df["DATASET"].values


#X, bağımsız değişkenli verilerdir, DATASET sütunu hariç hepsidir.
# Özelliklerden biri olarak dahil edilmesini istemediğimiz için etiket sütununu X'ten bırakılır
X = df.drop(labels = ["DATASET"], axis=1)  



#ADIM 4: VERİLER EĞİTİM VE TEST VERİLERİNE BÖLÜNÜR.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=61)
#random_state herhangi bir tamsayı olabilir ve veri kümesini rastgele bölmek için bir tohum olarak kullanılır.



#ADIM 5: MODELİN TANIMLANMASI VE VERİLERİN EĞİTİMİ.

from sklearn.linear_model import LogisticRegression   #Model içe aktarılır.
model = LogisticRegression(max_iter=1000)  #Modelin bir örneği oluşturulur.
model.fit(X_train, y_train)  # Eğitim verilerini kullanarak model eğitilir.



#ADIM 6: TEST VERİLERİ ÜZERİNDEN TAHMİN EDEREK MODEL TEST EDİLİR VE DOĞRULUK PUANI HESAPLANIR

prediction_test = model.predict(X_test)
from sklearn import metrics
print ("f1-score(f-measure) = ", metrics.f1_score(y_test,prediction_test))
print ("Precision(Kesinlik) = ", metrics.precision_score(y_test,prediction_test))
print ("Accuracy(Doğruluk)  = ", metrics.accuracy_score(y_test, prediction_test)) #Tahmin doğruluğu yazdırılır.



#ADIM 7: HANGİ DEĞİŞKENLERİN SONUÇ ÜZERİNDE EN FAZLA ETKİSİ OLDUĞUNA BAKILIR
# Tüm değişkenlerin ağırlıkları alınır.

print(model.coef_) #Her bağımsız değişken için katsayıları yazdırılır. 
weights = pd.Series(model.coef_[0], index=X.columns.values)

print("Her degisken icin agirliklar asagidaki gibidir...")
print(weights)

#+ DEĞER, DEĞİŞKENİN OLUMLU BİR ETKİSİ OLDUĞUNU GÖSTERİR.