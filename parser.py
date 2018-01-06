# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:28:02 2017

@author: dongg
"""

import findspark
findspark.init('C:/opt/spark/spark-2.2.0-bin-hadoop2.7')
import numpy as np
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import Row
import csv
import pandas as pd
from pyspark import SparkConf
from pyspark.sql.types import *

conf = SparkConf().setAppName("EDM").setMaster("local[2]")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
stu_log_1 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_1.csv')
stu_log_2 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_2.csv')
stu_log_3 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_3.csv')
stu_log_4 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_4.csv')
stu_log_5 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_5.csv')
stu_log_6 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_6.csv')
stu_log_7 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_7.csv')
stu_log_8 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_8.csv')
stu_log_9 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_9.csv')
training_label=sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('training_label.csv')

raw_data=stu_log_1.union(stu_log_2).union(stu_log_3).union(stu_log_4).union(stu_log_5).union(stu_log_6).union(stu_log_7).union(stu_log_8).union(stu_log_9)
#print((raw_data.count(), len(raw_data.columns)))
clean_data=raw_data.na.drop()
#print((clean_data.count(), len(clean_data.columns)))
clean_data=clean_data.dropDuplicates()
drop_data=clean_data.drop('confidence(GAMING)','confidence(OFF TASK)', 'confidence(FRUSTRATED)', 'confidence(CONCENTRATING)', 'confidence(BORED)','confidence(CONFUSED)','sumTime3SDWhen3RowRight', 'prev5count', 'sumRight','SY ASSISTments Usage','actionId','problemId','assignmentId','assistmentId','startTime','endTime','problemType','skill')

print((drop_data.count(), len(drop_data.columns)))
"""
#print (d1.first())

field = [StructField('ITEST_id',StringType(), True), StructField('AveKnow', StringType(), True),
         StructField('AveCarelessness', StringType(), True), StructField('AveCorrect', StringType(), True),
         StructField('NumActions', StringType(), True), StructField('AveResBored', StringType(), True),
         StructField('AveResEngcon', StringType(), True), StructField('AveResConf', StringType(), True),
         StructField('AveResFrust', StringType(), True), StructField('AveResOfftask', StringType(), True),
         StructField('AveResGaming', StringType(), True), StructField('timeTaken', StringType(), True),
         StructField('correct', StringType(), True),StructField('original', StringType(), True), 
         StructField('hint', StringType(), True),
         StructField('hintCount', StringType(), True), StructField('hintTotal', StringType(), True),
         StructField('scaffold', StringType(), True), StructField('bottomHint', StringType(), True),
         StructField('attemptCount', StringType(), True), StructField('frIsHelpRequest', StringType(), True),
         StructField('frPast5HelpRequest', StringType(), True), StructField('frPast8HelpRequest', StringType(), True),
         StructField('stlHintUsed', StringType(), True), StructField('past8BottomOut', StringType(), True),
         StructField('totalFrPercentPastWrong', StringType(), True), StructField('totalFrPastWrongCount', StringType(), True),
         StructField('frPast5WrongCount', StringType(), True), StructField('frPast8WrongCount', StringType(), True),
         StructField('totalFrTimeOnSkill', StringType(), True), StructField('timeSinceSkill', StringType(), True),
         StructField('frWorkingInSchool', StringType(), True), StructField('totalFrAttempted', StringType(), True),
         StructField('totalFrSkillOpportunities', StringType(), True), StructField('responseIsFillIn', StringType(), True),
         StructField('responseIsChosen', StringType(), True), StructField('endsWithScaffolding', StringType(), True),
         StructField('endsWithAutoScaffolding', StringType(), True), StructField('frTimeTakenOnScaffolding', StringType(), True),
         StructField('frTotalSkillOpportunitiesScaffolding', StringType(), True), StructField('totalFrSkillOpportunitiesByScaffolding', StringType(), True),
         StructField('frIsHelpRequestScaffolding', StringType(), True), StructField('timeGreater5Secprev2wrong', StringType(), True),
         StructField('helpAccessUnder2Sec', StringType(), True), StructField('timeGreater10SecAndNextActionRight', StringType(), True),
         StructField('consecutiveErrorsInRow', StringType(), True), StructField('sumTimePerSkill', StringType(), True),
         StructField('totalTimeByPercentCorrectForskill', StringType(), True), StructField('timeOver80', StringType(), True),
         StructField('manywrong', StringType(), True), StructField('RES_BORED', StringType(), True),
         StructField('RES_CONCENTRATING', StringType(), True),StructField('RES_CONFUSED', StringType(), True),
         StructField('RES_FRUSTRATED', StringType(), True),StructField('RES_OFFTASK', StringType(), True),
         StructField('RES_GAMING', StringType(), True),StructField('Ln-1', StringType(), True),StructField('Ln', StringType(), True)]
schema = StructType(field)
median_data = sqlContext.createDataFrame(sc.emptyRDD(), schema)
"""
"""
def filter_id(data, student_id, newCollection):
    filtered_df=data.where(data['ITEST_id'] == student_id)
    f_df_pandas=filtered_df.toPandas()
    mean_df=f_df_pandas.mean()
    v=mean_df.to_frame()
    r=v.transpose()
    newCollection.append(r)
    return newCollection
    
def write_to_csv(log, train, collection=[]):
    stu_id=train['ITEST_id']
    stu_id=stu_id.toPandas()
    ids=stu_id.values.tolist()
    for i in ids:
        collection=filter_id(log, i, collection)
    newData=pd.concat(collection)
    newData.to_csv('new.csv', encoding='utf-8', index=False)

write_to_csv(drop_data, training_label, collection=[])   
"""

#f1=drop_data.where(drop_data['ITEST_id'] == 8)
#print((f1.count(), len(f1.columns)))
"""   
if __name__ == '__main__':
    conf = SparkConf().setAppName("EDM").setMaster("local[2]")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    stu_log_1 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_1.csv')
    stu_log_2 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_2.csv')
    stu_log_3 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_3.csv')
    stu_log_4 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_4.csv')
    stu_log_5 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_5.csv')
    stu_log_6 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_6.csv')
    stu_log_7 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_7.csv')
    stu_log_8 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_8.csv')
    stu_log_9 = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('student_log_9.csv')
    training_label=sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('training_label.csv')
    raw_data=stu_log_1.union(stu_log_2).union(stu_log_3).union(stu_log_4).union(stu_log_5).union(stu_log_6).union(stu_log_7).union(stu_log_8).union(stu_log_9)
    clean_data=raw_data.na.drop()
    clean_data=clean_data.dropDuplicates()
    drop_data=clean_data.drop('confidence(GAMING)','confidence(OFF TASK)', 'confidence(FRUSTRATED)', 'confidence(CONCENTRATING)', 'confidence(BORED)','confidence(CONFUSED)','sumTime3SDWhen3RowRight', 'prev5count', 'sumRight','SY ASSISTments Usage','actionId','problemId','assignmentId','assistmentId','startTime','endTime','problemType','skill')
    write_to_csv(drop_data, training_label, collection=[])
    
    print("check new data at intelligent learning file")
"""

    