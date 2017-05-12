pyspark
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.sql import Row
from pyspark.sql import functions as sql_functions
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


raw_data = sqlContext.read.load('YearPredictionMSD.txt', 'text')
num_points = raw_data.count()

attribute_description = "90 attributes, 12 = timbre average, 78 = timbre covariance. \nThe first value is the year (target), ranging from 1922 to 2011. \nFeatures extracted from the 'timbre' features from The Echo Nest API. \nWe take the average and covariance over all 'segments', each segment being described by a 12-dimensional timbre vector."

df = raw_data.rdd.map(lambda row: str(row['value']).split(",")).map(lambda row: LabeledPoint(row[0], [float(x) for x in row[1:]])).toDF(["Features", "Year"])

print attribute_description
print '\nNumber of data points: ', num_points, "\n"

raw_data.take(1)


%pyspark

year_data = df.select("Year").groupBy("Year").count()
year_data.show()


%pyspark
year_data = year_data.toPandas()
year_data

%pyspark
from matplotlib.cm import get_cmap


plt.plot(year_data["Year"], year_data["count"], 'g.')
plt.xlabel("Year")
plt.ylabel("No. of songs")
plt.title("Number of songs by year")
plt.plot()

%pyspark
avg_year = year_data["Year"]
average_year = sum(avg_year)/len(avg_year)

print("Average year: ", average_year)

%pyspark

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol = 'prediction')

weights = [.8, .1, .1]
seed = 42
parsed_train_data_df, parsed_val_data_df, parsed_test_data_df = df.randomSplit(weights, seed= seed)

parsed_train_data_df.cache()
parsed_val_data_df.cache()
parsed_test_data_df.cache()
n_train = parsed_train_data_df.count()
n_val = parsed_val_data_df.count()
n_test = parsed_test_data_df.count()

print 'Training dataset size: {0}'.format(n_train)
print 'Validation dataset size: {0}'.format(n_val)
print 'Testing dataset size: {0}'.format(n_test)

%pyspark

# We use the average year as the prediction for all rows in the test data, and calculate the baseline error

preds_and_labels_test = parsed_test_data_df.rdd.map(lambda row: (float(1967), float(row['Year'])))
preds_and_labels_test_df = sqlContext.createDataFrame(preds_and_labels_test, ["prediction", "label"])
rmse_test_base = evaluator.evaluate(preds_and_labels_test_df)

print 'Baseline Test RMSE = {0:.3f}'.format(rmse_test_base)

%pyspark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf


# Linear regression model parameter values
num_iters = 500  # iterations
reg = 1e-1  # regParam
alpha = .2  # elasticNetParam
use_intercept = True  # intercept

# parsed_train_data_df = parsed_train_data_df.withColumn("Year", parsed_train_data_df["Year"].cast(DoubleType()))
parsed_train_data_df = parsed_train_data_df.rdd.map(lambda row: (Vectors.dense(row["Features"]), float(row['Year'])))
parsed_train_data_df = sqlContext.createDataFrame(parsed_train_data_df,["features","label"])
parsed_train_data_df
lin_reg = LinearRegression(maxIter = num_iters, regParam = reg, elasticNetParam = alpha, fitIntercept = use_intercept, labelCol = 'label', featuresCol = 'features')

first_model = lin_reg.fit(parsed_train_data_df)

%pyspark

coeffs_LR1 = first_model.coefficients
intercept_LR1 = first_model.intercept
print coeffs_LR1, intercept_LR1

%pyspark

parsed_val_data_df = parsed_val_data_df.rdd.map(lambda row: (Vectors.dense(row["Features"]), float(row['Year'])))
parsed_val_data_df = sqlContext.createDataFrame(parsed_val_data_df,["features","label"])

#parsed_val_data_df = parsed_val_data_df.withColumn("label", parsed_val_data_df["label"].cast(DoubleType()))
val_pred_df = first_model.transform(parsed_val_data_df)
rmse_val_LR1 = evaluator.evaluate(val_pred_df)

print ('Validation RMSE:LR1 = ',  rmse_val_LR1)

%pyspark
model_pred_label = val_pred_df.toPandas()

%pyspark
plt.plot(model_pred_label["label"][:200000], ".")
plt.plot(model_pred_label["prediction"][:200000], "r.")
