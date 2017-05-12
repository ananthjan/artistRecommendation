%pyspark
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.sql import Row
from pyspark.sql import functions as sql_functions
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyspark.mllib.recommendation import *
import random
from operator import *
from collections import defaultdict

%pyspark

artistData=sc.textFile('artist_data_small.txt')

artistAlias=sc.textFile('artist_alias_small.txt')

userArtistData=sc.textFile('user_artist_data_small.txt')

artistData=artistData.map(lambda x: x.split("\t")).map(lambda x: (int(x[0]), x[1]))
artistAlias=artistAlias.map(lambda x: x.split("\t")).map(lambda x: (int(x[0]), int(x[1])))
userArtistData = userArtistData.map(lambda x: x.split(" ")).map(lambda x: (int(x[0]), int(x[1]), int(x[2])))


artistAliasDict =  dict(artistAlias.collect())

userArtistData = userArtistData.map(lambda x: (x[0], artistAliasDict[x[1]], x[2]) if x[1] in artistAliasDict.keys() else x)


trainData, validationData, testData = userArtistData.randomSplit([8,1,1], seed=11)
trainData.cache()

model = ALS.trainImplicit(trainData, rank=10, seed=345)

recommendations = model.recommendProducts(2102019, 10)
recommendations

predictions = map(lambda it: it.product, recommendations)

artistNames = dict(artistData.collect())

for key in predictions:
     print artistNames[key]
    
%pyspark

df = df = sqlContext.createDataFrame(userArtistData, ['userId', 'ArtistId', 'count'])
df = df.toPandas()

user = df[df["userId"] == 2102019]
user_favs = df[["ArtistId", "count"]]
user_favs = user_favs.sort(["count"], ascending=False)

top20 = user_favs.head(20)[2:12]
artistNames = dict(artistData.collect())
cc = top20["count"]
artId = top20["ArtistId"].values
# cc.values
names = []
for i in artId:
    names.append(artistNames[i])
# names
plt.plot(cc.values, "rp-")
plt.xticks(range(len(names)), names)
plt.xticks(rotation=20)
plt.plot()

%pyspark
u = user_favs.sort(["count"], ascending=False)
u.head(20)[:-10]
