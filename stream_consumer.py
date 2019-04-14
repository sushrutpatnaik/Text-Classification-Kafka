from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, regexp_replace
from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer,StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from kafka import KafkaConsumer


spark = SparkSession \
        .builder \
        .appName("") \
        .getOrCreate()


accuracy_list_LR = []
precision_list_LR = []
fmeasure_list_LR = []
recall_list_LR = []

accuracy_list_DT = []
precision_list_DT = []
fmeasure_list_DT = []
recall_list_DT = []


classifierNames = [
            "Decision Tree",
            "Logistic Regression"
        ]

def main():
    if len(sys.argv) != 3:
        print("Usage: MLModel <file>", file=sys.stderr)
        sys.exit(-1)

    #reads from news articles from text file
    trainlines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    traindata = preprocess(trainlines)
    traindata.toPandas()
    modelList,nameList= classify(traindata)

    consumer = KafkaConsumer('guardian2', bootstrap_servers=['localhost:9092'], consumer_timeout_ms=3000)
    consume(modelList, nameList, "test.txt", consumer)
    spark.stop()



def preprocess(lines):
    # strips " from each line and splits the label and the text
    lineSplit = lines.map(lambda x: x.replace("\\u", " ")).map(lambda x: x.strip('"')).map(lambda x: x.split('||'))

    # converts into a Dataframe
    lineDF = spark.createDataFrame(lineSplit).toDF("label", "sentence")
    cleanDF = lineDF.select('label', (regexp_replace('sentence', "\\d+", "")).alias('sentence'))


    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

    regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")

    countTokens = udf(lambda words: len(words), IntegerType())

    tokenized = tokenizer.transform(cleanDF)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    removeWords = remover.transform(tokenized)


    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1500)
    featurizedData = hashingTF.transform(removeWords)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)


    allData = rescaledData.select(
        [c for c in rescaledData.columns if c not in {'sentence', 'words', 'filtered', 'rawFeatures'}])

    finalDF = allData.withColumn("label", allData["label"].cast(DoubleType()))
    return(finalDF)


def classify(traindata):
    modelList = []
    nameList = []
    for name in classifierNames:
        if name == "Decision Tree":
            classifier = DecisionTreeClassifier(labelCol="label", featuresCol="features")

        elif name == "Logistic Regression":
            classifier = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)

        print("The Classifier  is : ", str(name))

        print("Training Data set Count: " + str(traindata.count()))
        print("\n")

        model = classifier.fit(traindata)
        modelList.append(model)
        nameList.append(str(name))

    return modelList, nameList


def predict(modelList, nameList, testdata):
    index = 0
    for models in modelList:
        print("Model is  ", nameList[index])
        print("\n")
        predictions = models.transform(testdata)

        # Select example rows to display.
        predictions.select("prediction", "label", "features").show()

        # Select (prediction, true label) and compute test error
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

        accuracy = evaluator.evaluate(predictions)
        prediction_RDD = predictions.select(['label', 'prediction']) \
            .rdd.map(lambda line: (line[1], line[0]))

        metrics = MulticlassMetrics(prediction_RDD)

        if nameList[index] == "Logistic Regression":
            accuracy_list_LR.append(accuracy)
            fmeasure_list_LR.append(metrics.fMeasure())
            precision_list_LR.append(metrics.precision())
            recall_list_LR.append(metrics.recall())

        elif nameList[index] == "Decision Tree":
            accuracy_list_DT.append(accuracy)
            fmeasure_list_DT.append(metrics.fMeasure())
            precision_list_DT.append(metrics.precision())
            recall_list_DT.append(metrics.recall())

        index = index + 1


def consume(modelList, nameList, fileName, consumer):

    outputFile = open(fileName, "w")
    count = 0
    for msg in consumer:
        if count != 15:
            outputFile.write((msg.value).decode("utf-8") +"\n")
            count = count + 1
        elif count == 15:
            testlines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
            testdata = preprocess(testlines)
            testdata.toPandas()
            predict(modelList, nameList, testdata)
            count = 0
            outputFile.close()
            outputFile = open(fileName, "w")

    printMetrics()
    outputFile.close()


def printMetrics():

    print("Results for Decision Tree Model ")
    dt_accuracy = sum(accuracy_list_DT)
    dt_fmeasure = sum(fmeasure_list_DT)
    dt_precision = sum(precision_list_DT)
    dt_recall = sum(recall_list_DT)

    print("Accuracy= %g " % (dt_accuracy / len(accuracy_list_DT)))
    print("Precision = %s " % (dt_precision / len(precision_list_DT)))
    print("Recall = %s " % (dt_recall / len(recall_list_DT)))
    print("F Measure = %s " % (dt_fmeasure / len(fmeasure_list_DT)))

    print("Results for Logistic Regression Model ")
    lr_accuracy = sum(accuracy_list_LR)
    lr_fmeasure = sum(fmeasure_list_LR)
    lr_precision = sum(precision_list_LR)
    lr_recall = sum(recall_list_LR)

    print("Accuracy= %g " % (lr_accuracy / len(accuracy_list_LR)))
    print("Precision = %s " % (lr_precision / len(precision_list_LR)))
    print("Recall = %s " % (lr_recall / len(recall_list_LR)))
    print("F Measure = %s " % (lr_fmeasure / len(fmeasure_list_LR)))
    print("\n")


if __name__ == "__main__":
    main()
