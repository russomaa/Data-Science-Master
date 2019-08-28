# -*- coding: utf-8 -*-
"""
Sistema de Recomendación (Filtrado Colaborativo)

- Escribir una implementación del mismo para Spark
- Evaluar el modelo de filtrado colaborativo sobre el dataset de películas (Movies)

Big Data, Máster Ciencia de Datos
Mayra Russo Botero 
"""



SEED=666

# libraries
import numpy as np
np.random.seed(SEED)
import pandas as pd
import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext,SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.sql.functions import col,min,max,avg

# check if spark context is defined
sc = SparkContext()

# initialize SQLContext
sqlContext = SQLContext(sc)

# initialize Spark Session
spark = SparkSession.builder.appName('rec_app').getOrCreate()

# data import
#users data
u_cols = 'user_id::sex::age::occupation::zip_code'.split('::')
df_users = sqlContext.read.csv('data/users.dat', sep=':',header=False,
                               inferSchema=True)
#ratings data
r_cols = 'userID::movieID::rating::timestamp'.split('::')
df_ratings = sqlContext.read.csv('data/ratings.dat', sep=':',header=False,
                                 inferSchema=True)
#movies data
m_cols = 'movie_id::title::genres'.split('::')
df_movies = sqlContext.read.csv('data/movies.dat', sep=':',header=False,
                               inferSchema=True)

#arrange dataframes 
#extract correct columns and rename them, plus add headers 
#ratings df 
dis_columns = np.array(df_ratings.columns)
dis_columns = dis_columns[range(0, len(df_ratings.columns), 2)]
df_ratings = df_ratings.select(dis_columns.tolist())

assert len(r_cols) == len(dis_columns)
for i in range(len(r_cols)):
    df_ratings = \
        df_ratings.withColumnRenamed(dis_columns[i], r_cols[i])
df_ratings.printSchema()

#movies df 
dis_columns1 = np.array(df_movies.columns)
dis_columns1 = dis_columns1[range(0, len(df_movies.columns), 2)]
df_movies = df_movies.select(dis_columns1.tolist())

assert len(m_cols) == len(dis_columns1)
for i in range(len(m_cols)):
    df_movies = \
        df_movies.withColumnRenamed(dis_columns1[i], m_cols[i])
df_movies.printSchema()

#users df 
dis_columns2 = np.array(df_users.columns)
dis_columns2 = dis_columns2[range(0, len(df_users.columns), 2)]
df_users = df_users.select(dis_columns2.tolist())

assert len(u_cols) == len(dis_columns2)
for i in range(len(u_cols)):
    df_users = \
        df_users.withColumnRenamed(dis_columns2[i], u_cols[i])
df_users.printSchema()

#data exploration
df_ratings.show()
df_movies.show()
df_users.show()

#extracting distinct user ids and movies, sparsity 
#number of ratings in matrix
numerator = df_ratings.count()
#distinct users and movies 
users=df_ratings.select('userID').distinct().count()
movies=df_ratings.select('movieID').distinct().count()
#number of ratings matrix could contain if no empty cells 
denominator = users * movies 

#calculating sparsity
sparsity = 1 - (numerator*1.0 / denominator)
sparsity

#removes users with less than 20 ratings
df_ratings.groupBy("userID").count().filter(col("count") >= 20).show()

# max and min num of ratings by userId
df_ratings.groupBy("userID").count().select(min("count")).show()
df_ratings.groupBy("userID").count().select(max("count")).show()


#fitting the model 
#we will use a smaller dataset 
(small_ratings,everything)= df_ratings.randomSplit([0.0010,0.999],seed=SEED)
# split data
(training_data, test_data) = small_ratings.randomSplit([0.8, 0.2],seed=SEED)
# ALS model
als = ALS(userCol="userID", itemCol="movieID", ratingCol="rating", nonnegative=True,
coldStartStrategy="drop", implicitPrefs=False)


#create a ParamGridBuilder 
param_grid = ParamGridBuilder()\
            .addGrid(als.rank,[10, 50]) \
            .addGrid(als.maxIter,[5]) \
            .addGrid(als.regParam,[.05, .1, 1.5]) \
            .build()
                      
#how to evaluate performance
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
predictionCol="prediction")

#create cross validator
cv = CrossValidator(estimator = als,
                    estimatorParamMaps = param_grid,
                    evaluator = evaluator,
                    numFolds = 2)

# Run the cv on the training data
model = cv.fit(training_data)
# Extract best combination of values from cross validation
best_model = model.bestModel

# Generate test set predictions and evaluate using RMSE
predictions = best_model.transform(test_data)
rmse = evaluator.evaluate(predictions)
# Print evaluation metrics and model parameters
print ("**Best Model**")
print ("RMSE = " , rmse)
print (" Rank: ", best_model.rank)
print (" MaxIter: ", best_model._java_obj.parent().getMaxIter())
print (" RegParam: ", best_model._java_obj.parent().getRegParam())

##### 
predictions.show()
test_data.count()
predictions.count()
