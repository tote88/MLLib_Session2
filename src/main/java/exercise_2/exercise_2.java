package exercise_2;


import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.types.DataTypes.*;


public class exercise_2 {

	public static void pacientesModel(SparkSession ss) {

		JavaSparkContext jsc = new JavaSparkContext(ss.sparkContext());

		Dataset<Row> pacientes = getRowDataset(ss);

		pacientes.show();

	    ss.stop(); 
	}

	private static Dataset<Row> getRowDataset(SparkSession ss) {
		return ss.read()
					.format("com.databricks.spark.csv")
					.option("header", "true")
					.option("inferSchema", "true")
					.option("sep", ";")
					//.schema(schema)
					.csv("C:\\Users\\tote8\\Documents\\MBDMA\\HandsOn\\MLLib\\mllib2\\src\\main\\resources\\PacientesSim.csv");
	}
}
