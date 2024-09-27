import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

# 创建 Spark 会话
spark = SparkSession.builder \
    .appName("MLflow Spark Distributed Example") \
    .getOrCreate()

# 设置实验
mlflow.set_experiment("Distributed Spark ML Experiment")

# 加载数据
data_path = "path/to/your/data.csv"  # 更改为实际数据路径
data = spark.read.csv(data_path, header=True, inferSchema=True)

# 特征工程
assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features")  # 更改为实际特征列
data = assembler.transform(data)

# 划分数据集
(train_data, test_data) = data.randomSplit([0.8, 0.2])

# 开始 MLflow 运行
with mlflow.start_run():
    # 定义模型
    rf = RandomForestRegressor(featuresCol="features", labelCol="label")

    # 训练模型（分布式）
    model = rf.fit(train_data)

    # 记录模型
    mlflow.spark.log_model(model, "spark-model")

    # 预测
    predictions = model.transform(test_data)

    # 评估模型
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    # 记录评估指标
    mlflow.log_metric("rmse", rmse)

    # 记录参数
    mlflow.log_param("numTrees", rf.getNumTrees)
    mlflow.log_param("maxDepth", rf.getMaxDepth)

    print(f"Root Mean Squared Error (RMSE): {rmse}")

# 停止 Spark 会话
spark.stop()
