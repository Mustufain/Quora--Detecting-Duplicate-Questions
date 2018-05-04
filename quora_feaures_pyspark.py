from pyspark import SparkContext
from pyspark import SQLContext
import logging
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec,Tokenizer,Normalizer,VectorAssembler,StopWordsRemover
from functools import reduce
from pyspark.sql.types import StringType,IntegerType,FloatType
import sys
from fuzzywuzzy import fuzz
from scipy import spatial
from scipy import stats
from pyspark import Row
from pyspark.mllib.linalg import Vectors
import time
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder,TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import SQLTransformer
from scipy import spatial
from scipy import stats
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import math
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml import Pipeline, PipelineModel
import numpy as np

t0 = time.time()
reload(sys)
sys.setdefaultencoding('utf-8')
sc=SparkContext('local','untitled1')


sc.setLogLevel("ERROR")
log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)


# create a file handler
handler = logging.FileHandler('error.log')
handler.setLevel(logging.ERROR)

#
# # create a logging format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
#
#
# # add the handlers to the logger
#LOGGER.addHandler(handler)



sqlContext = SQLContext(sc)


data = sqlContext.read.format("com.databricks.spark.csv").\
        option("header", "true").\
        option("inferschema", "true").\
        option("mode", "DROPMALFORMED").\
        option("encoding", "UTF-8") .\
        load("data/train.csv")



data = data.withColumnRenamed("is_duplicate", "label")
data = data.withColumn("label", data.label.cast(IntegerType()))
data=data.fillna(" ")

##  subset origional data to run some tests
rnd_seed = 42
fractions = {0.0: 0.025, 1.0: 0.025}
subset = data.stat.sampleBy('label', fractions, seed=rnd_seed)
data.groupBy('label').count().show()
subset.groupBy('label').count().show()
########

#train, test = data.randomSplit([0.9, 0.1], seed=12345) # splitting for validation
train, test = subset.randomSplit([0.9, 0.1], seed=12345) # splitting for validation



######################################################################################

remove_punctuation = F.udf(lambda row:"".join([s if s.isalpha() or s.isdigit() or s=="'" or s==" " else ' ' for s in row]),StringType())
train = (reduce(
     lambda memo_df, col_name: memo_df.withColumn(col_name, remove_punctuation(col_name)),
     ["question1", "question2"],
     train
 ))

########################## GENERATING UDF TO BE USED IN PIPELINE #############################

def common_words(q1, q2):
    return len(list(set(q1).intersection(q2)))


def total_words(q1, q2):
    return max(len(q1), len(q2))


def diff_two_lengths(lengthq1, lengthq2):
    return abs(lengthq1 - lengthq2)


def fuzz_qratio(q1,q2):

    return fuzz.QRatio(q1, q2)


def fuzz_partialratio(q1,q2):

    return fuzz.partial_ratio(q1,q2)


def fuzz_partial_token_setratio(q1,q2):
    return fuzz.partial_token_set_ratio(q1,q2)


def fuzz_partial_token_sortratio(q1,q2):
    return fuzz.partial_token_sort_ratio(q1,q2)


def fuzz_token_setratio(q1,q2):
    return fuzz.token_set_ratio(q1,q2)


def fuzz_token_sortratio(q1,q2):
    return fuzz.token_sort_ratio(q1,q2)


def fuzz_wratio(q1,q2):
    return fuzz.WRatio(q1,q2)



textlength_q1 = sqlContext.udf.register(
"textlength_q1",
lambda row : len(row),
   "integer"
)


transformer_textlength_q1=SQLTransformer(
    statement = "SELECT *, textlength_q1(question1_tokens_filtered) textlengthq1 FROM __THIS__"
)

textlength_q2 = sqlContext.udf.register(
    "textlength_q2",
    lambda row : len(row),
    "integer"
)

transformer_textlength_q2=SQLTransformer(
    statement = "SELECT *, textlength_q2(question2_tokens_filtered) textlengthq2 FROM __THIS__"
)

udf_commonwords = sqlContext.udf.register(
    'common_words',
    common_words,
    'integer'
)

transformer_commonwords=SQLTransformer(
    statement = "SELECT *, common_words(question1_tokens_filtered,question2_tokens_filtered) commonwords FROM __THIS__"
)

udf_totalwords = sqlContext.udf.register(
    'totalwords',
    total_words,
    'integer'
)

transformer_totalwords=SQLTransformer(
    statement = "SELECT *, totalwords(question1_tokens_filtered,question2_tokens_filtered) totalwords FROM __THIS__"
)

udf_difftwolength = sqlContext.udf.register(
    'diff_two_lengths',
    diff_two_lengths,
    'integer'
)


transformer_difftwolength=SQLTransformer(
    statement = "SELECT *, diff_two_lengths(textlengthq1,textlengthq2) difftwolength FROM __THIS__"
)


udf_fuzzqratio = sqlContext.udf.register(
    'fuzz_qratio',
    fuzz_qratio,
    'integer'
)


transformer_fuzz_qratio=SQLTransformer(
    statement = "SELECT *, fuzz_qratio(question1,question2) fuzz_qratio FROM __THIS__"
)

udf_fuzzpartialratio = sqlContext.udf.register(
    'fuzz_partialratio',
    fuzz_partialratio,
    'integer'
)


transformer_fuzz_partialratio=SQLTransformer(
    statement = "SELECT *, fuzz_partialratio(question1,question2) fuzz_partialratio FROM __THIS__"
)



udf_fuzz_partial_token_setratio = sqlContext.udf.register(
    'fuzz_partial_token_setratio',
    fuzz_partial_token_setratio,
    'integer'
)


transformer_fuzz_partial_token_setratio=SQLTransformer(
    statement = "SELECT *, fuzz_partial_token_setratio(question1,question2) fuzz_partial_token_setratio FROM __THIS__"
)


udf_fuzz_partial_token_sortratio = sqlContext.udf.register(
    'fuzz_partial_token_sortratio',
    fuzz_partial_token_sortratio,
    'integer'
)


transformer_fuzz_partial_token_sortratio=SQLTransformer(
    statement = "SELECT *, fuzz_partial_token_sortratio(question1,question2) fuzz_partial_token_sortratio FROM __THIS__"
)


udf_fuzz_token_setratio = sqlContext.udf.register(
    'fuzz_token_setratio',
    fuzz_token_setratio,
    'integer'
)


transformer_fuzz_token_setratio=SQLTransformer(
    statement = "SELECT *, fuzz_token_setratio(question1,question2) fuzz_token_setratio FROM __THIS__"
)


udf_fuzz_token_sortratio = sqlContext.udf.register(
    'fuzz_token_sortratio',
    fuzz_token_sortratio,
    'integer'
)


transformer_fuzz_token_sortratio=SQLTransformer(
    statement = "SELECT *, fuzz_token_sortratio(question1,question2) fuzz_token_sortratio FROM __THIS__"
)


udf_fuzz_wratio = sqlContext.udf.register(
    'fuzz_wratio',
    fuzz_wratio,
    'integer'
)


transformer_fuzz_wratio=SQLTransformer(
    statement = "SELECT *, fuzz_wratio(question1,question2) fuzz_wratio FROM __THIS__"
)



manhattan = sqlContext.udf.register(
    'manhattan',
    lambda x, y: float(-5555555) if math.isnan(float(spatial.distance.cityblock(x, y))) or math.isinf(float(spatial.distance.cityblock(x, y))) else float(spatial.distance.cityblock(x, y)),
    'float'
)

transformer_manhattan=SQLTransformer(
    statement = "SELECT *, manhattan(q1_vectors,q2_vectors) manhattan FROM __THIS__"
)


euclidean = sqlContext.udf.register(
    'euclidean',
    lambda x, y: float(-5555555) if math.isnan(float(spatial.distance.euclidean(x, y))) or math.isinf(float(spatial.distance.canberra(x, y))) else float(spatial.distance.canberra(x, y)),
    'float'
)

transformer_euclidean=SQLTransformer(
    statement = "SELECT *, euclidean(q1_vectors,q2_vectors) euclidean FROM __THIS__"
)

minkowski = sqlContext.udf.register(
    'minkowski',
    lambda x, y: float(-5555555) if math.isnan(float(spatial.distance.minkowski(x, y))) or math.isinf(float(spatial.distance.canberra(x, y))) else float(spatial.distance.canberra(x, y)),
    'float'
)

transformer_minkowski=SQLTransformer(
    statement = "SELECT *, minkowski(q1_vectors,q2_vectors) minkowski FROM __THIS__"
)


canberra = sqlContext.udf.register(
    'canberra',
    lambda x, y: float(-5555555) if math.isnan(float(spatial.distance.canberra(x, y))) or math.isinf(float(spatial.distance.canberra(x, y))) else float(spatial.distance.canberra(x, y)),
    'float'
)

transformer_canberra=SQLTransformer(
    statement = "SELECT *, canberra(q1_vectors,q2_vectors) canberra FROM __THIS__"
)

braycurtis = sqlContext.udf.register(
    'braycurtis',
    lambda x, y: float(-5555555) if math.isnan(float(spatial.distance.braycurtis(x, y))) or math.isinf(float(spatial.distance.braycurtis(x, y))) else float(spatial.distance.braycurtis(x, y)) ,
    'float'
)

transformer_braycurtis=SQLTransformer(
    statement = "SELECT *, braycurtis(q1_vectors,q2_vectors) braycurtis FROM __THIS__"
)

jaccard = sqlContext.udf.register(
    'jaccard',
    lambda x, y: float(-5555555) if math.isnan(float(spatial.distance.jaccard(x, y))) or math.isinf(float(spatial.distance.jaccard(x, y))) else float(spatial.distance.jaccard(x, y)),
    'float'
)

transformer_jaccard=SQLTransformer(
    statement = "SELECT *, jaccard(q1_vectors,q2_vectors) jaccard FROM __THIS__"
)

cosine = sqlContext.udf.register(
    'cosine',
    lambda x, y: float(-5555555) if math.isnan(float(spatial.distance.cosine(x, y))) or math.isinf(float(spatial.distance.cosine(x, y))) else float(spatial.distance.cosine(x, y)),
    'float'
)

transformer_cosine=SQLTransformer(
    statement = "SELECT *, cosine(q1_vectors,q2_vectors) cosine FROM __THIS__"
)

kurtosis = sqlContext.udf.register(
    'kurtosis',
    lambda x: float(-5555555) if math.isnan(stats.kurtosis(x)) or math.isinf(stats.kurtosis(x)) else stats.kurtosis(x),
    'float'
)

transformer_kurtosis_q1=SQLTransformer(
    statement = "SELECT *, kurtosis(q1_vectors) kurtosis_q1 FROM __THIS__"
)

transformer_kurtosis_q2=SQLTransformer(
    statement = "SELECT *, kurtosis(q2_vectors) kurtosis_q2 FROM __THIS__"
)

skew = sqlContext.udf.register(
    'skew',
    lambda x: float(-5555555) if  math.isnan(stats.skew(x)) or math.isinf(stats.skew(x)) else stats.skew(x),
    'float'
)


transformer_skew_q1=SQLTransformer(
    statement = "SELECT *, skew(q1_vectors) skew_q1 FROM __THIS__"
)


transformer_skew_q2=SQLTransformer(
    statement = "SELECT *, skew(q2_vectors) skew_q2 FROM __THIS__"
)


###########################################################################################

token_q1=Tokenizer(inputCol='question1',outputCol='question1_tokens')  # converted to lower case implicitly
token_q2=Tokenizer(inputCol='question2',outputCol='question2_tokens') # converted to lower case implicitly
remover_q1=StopWordsRemover(inputCol='question1_tokens',outputCol='question1_tokens_filtered')
remover_q2=StopWordsRemover(inputCol='question2_tokens',outputCol='question2_tokens_filtered')


q1w2model = Word2Vec(inputCol='question1_tokens_filtered',outputCol='q1_vectors',vectorSize=300)
q1w2model.setSeed(1)

q2w2model = Word2Vec(inputCol='question2_tokens_filtered',outputCol='q2_vectors',vectorSize=300)
q2w2model.setSeed(1)


assembler = VectorAssembler(inputCols=["commonwords","totalwords","textlengthq1","textlengthq2",
                                       'difftwolength','fuzz_qratio','fuzz_partialratio','fuzz_partial_token_setratio',
                                       'fuzz_token_sortratio','fuzz_token_setratio','fuzz_wratio',
                                       'q1_vectors','q2_vectors','manhattan','euclidean','minkowski','canberra','braycurtis','jaccard',
                                       'cosine','kurtosis_q1','kurtosis_q2','skew_q1','skew_q2'],outputCol="features")

lr = LogisticRegression(labelCol='label',featuresCol='features', weightCol='class_weight',family='binomial')
######################################### HYPER PARAMETERS #########################

windowSize = range(5,10)
minCount = range(5,20)
maxIter= [10,100,1000]
regParam= [0.1,0.01]

# Total 24 Models

######################################################################################

pipeline=Pipeline(stages=[token_q1,token_q2,remover_q1,remover_q2,
                          transformer_textlength_q1,transformer_textlength_q2,transformer_totalwords,
                          transformer_commonwords,transformer_difftwolength,
                          transformer_fuzz_qratio,transformer_fuzz_partial_token_setratio,
                          transformer_fuzz_partial_token_sortratio,transformer_fuzz_token_setratio,
                          transformer_fuzz_token_sortratio,transformer_fuzz_partialratio,transformer_fuzz_wratio,
                          q1w2model,q2w2model,
                          transformer_manhattan, transformer_braycurtis, transformer_canberra,
                          transformer_cosine,transformer_euclidean,
                          transformer_jaccard,transformer_minkowski,transformer_kurtosis_q1,
                          transformer_kurtosis_q2,transformer_skew_q1,transformer_skew_q2,
                          assembler])

# paramGrid only takes list of values not integers

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam,regParam) \
    .build()
    # .addGrid(q1w2model.windowSize,windowSize) \
    # .addGrid(q1w2model.minCount,minCount) \
    # .addGrid(q2w2model.windowSize,windowSize) \
    # .addGrid(q2w2model.minCount,minCount) \
    # .addGrid(lr.maxIter,maxIter) \
    # .addGrid(lr.regParam, regParam) \
    #

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC')


tvs = TrainValidationSplit(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          trainRatio=0.8)

# Add class weight to class weights (imbalanced data set)

label1count = train.filter(F.col('label') == '1').count() #duplicate
datacount=train.count()
weight_balance = np.round(float(datacount-label1count)/float(datacount),decimals=2)

train = (train
         .withColumn('class_weight',
        F.when(F.col('label') == '1', F.lit(weight_balance)).otherwise(F.lit(1 - weight_balance))))
train.groupBy('label').count().show()
# stratified sample (train and validation subset)
fractions = {0.0: 0.25, 1.0: 0.25}
rnd_seed=42
train_pipeline_model =pipeline.fit(train)
assembled_train_df= train_pipeline_model.transform(train)
assembled_validation_df = assembled_train_df.stat.sampleBy('label', fractions, seed=rnd_seed).cache()

# subtract the validation set from the original training set to get 75% of the entire data
# and same distribution of income levels
assembled_train_df = assembled_train_df.subtract(assembled_validation_df).cache()
assembled_validation_df.groupBy('label').count().show()
assembled_train_df.groupBy('label').count().show()
model=lr.fit(assembled_train_df)
train_preds=model.transform(assembled_validation_df)
train_summary = model.evaluate(assembled_train_df)
validation_summary = model.evaluate(assembled_validation_df)
assembled_test_df= train_pipeline_model.transform(test)
test_preds = model.transform(assembled_test_df)
test_summary = model.evaluate(assembled_test_df)

print (train_summary.areaUnderROC)
print (validation_summary.areaUnderROC)
print (test_summary.areaUnderROC)

exit(0)
model = tvs.fit(train) # model is the model with combination of parameters that performed best
best_model=model.bestModel.stages[-1]
tvs_train_summary=best_model.summary

train_roc=tvs_train_summary.areaUnderROC
print train_roc
best_model.write().overwrite().save('models/logisticregression_best_model')
predictions =  model.transform(test)
test_roc= evaluator.evaluate(predictions)
print test_roc




#bestModel = PipelineModel.load('models/logisticregression_best_model')
