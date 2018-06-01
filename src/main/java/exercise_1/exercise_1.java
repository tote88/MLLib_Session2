package exercise_1;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import java.io.IOException;
import java.util.HashMap;

import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.types.DataTypes.*;

import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;


public class exercise_1 {

	// Crea un hashmap a partir del fichero de pares (ID_Mail, label)
	public static HashMap<String, Integer> loadSampleLabels2(String filePath)
	{
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		File f = new File(filePath);
		try {
			List<String> lines = FileUtils.readLines(f);
			for (String line : lines) {
				String[] split = line.split(",");
				map.put(split[0], Integer.valueOf(split[1]));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return(map);
	}

	// Lee sobre un Dataset con dos variables los mail files <Filename,Content>
	public static Dataset<Row> readMailFiles(SparkSession ss,JavaSparkContext jsc, String path)
	{
		
		Dataset<Row> emailsDataset;
		// ToDo. Completar el código
		JavaRDD<Row> textFiles = jsc.wholeTextFiles(path).map(x -> RowFactory.create(x._1,x._2));
		StructType schema = new StructType(new StructField[]
				{
						createStructField("fileName", StringType, true),
						createStructField("content", StringType, true)
				});
        emailsDataset = ss.createDataFrame(textFiles, schema);
		return(emailsDataset);
	}

	// Transforma el vector de palabras al modelo TF_IDF
	public static Dataset<Row> transformTFIDF (Dataset<Row> ds, int numFeatures)
	{
	    Dataset<Row> featurizedData = null;
		Tokenizer tokenizer = new Tokenizer()
				.setInputCol("content")
				.setOutputCol("words");
		Dataset<Row> tokenized = tokenizer.transform(ds);

		HashingTF hashingTF = new HashingTF()
				.setNumFeatures(numFeatures)
				.setInputCol(tokenizer.getOutputCol())
				.setOutputCol("features");

		featurizedData = hashingTF.transform(tokenized);

		return featurizedData;
	}

	// Ajusta un modelo SVM lineal mediante CV seleccionando el mejor parámetro C
	public static CrossValidatorModel fitModel(Dataset<Row> train)
	{
		LinearSVC linearSVC = new LinearSVC()
				.setMaxIter(10)
				.setLabelCol("label")
				.setFeaturesCol("features");

		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(linearSVC.regParam(), new double[]{0.01, 0.1, 0.2, 0.5, 1})
				.build();

		CrossValidator cv = new CrossValidator()
				.setEvaluator(new BinaryClassificationEvaluator())
				.setEstimator(linearSVC)
				.setEstimatorParamMaps(paramGrid);

		CrossValidatorModel cvModel = cv.fit(train);
		return cvModel;
	}
	    
	
	public static void spamDetection(SparkSession ss) {


		// leemos el fichero de nombres y labels con formato (fileId,label) y creamos un hasmap
		HashMap<String, Integer> labels = loadSampleLabels2("src/main/resources/1_spam-mail.tr.label");

		// Obtenemos el spark context
		JavaSparkContext jsc = new JavaSparkContext(ss.sparkContext());

		// Utilizamos broadcast para enviar de forma eficiente el hashmap a los workers
		final Broadcast<HashMap<String, Integer>> bc = jsc.broadcast(labels);

		// Leemos sobre un dataset todos los ficheros de train 
		Dataset<Row> eMails = readMailFiles(ss, jsc, "src\\main\\resources\\1_TR");
	 	
		// Definimos una UDF (User Defined Function) para obtener el label en el 
		// hashmap a partir del nombre del fichero
		ss.udf().register("getLabel", new UDF1<String, Integer>() {
	 	      public Integer call(final String fileName) {
	 	    	    Pattern pattern = Pattern.compile(".*TRAIN_([0-9]*).*");
	 				Matcher mClass = pattern.matcher(fileName);
	 				int label = 0;
	 				if (mClass.find())
	 	            {
	 	            	String mailID = mClass.group(1);
	 	            	label = Integer.valueOf(bc.value().get(mailID));
	 	            }
	 	        return (label);
	 	      }
	 	    }, IntegerType);

		
		// Añadimos una nueva columna label
	    Dataset<Row> labeledMails = eMails.withColumn("label", callUDF("getLabel", eMails.col("fileName")));

	    // Aplicamos al texto del dataset el modelo TD_IDF para obtener una nueva variable
	    // "features" que contiene el vector TF_IDF
	    Dataset<Row> featurizedData = transformTFIDF (labeledMails, 1000);

	    // Dividimos el dataset en train i test	    
	    // ToDo.Completar el código

		Dataset<Row>[] splittedDatasets = featurizedData.randomSplit(new double[]{0.7, 0.3}, 1001);
		Dataset<Row> train = splittedDatasets[0];
		Dataset<Row> test = splittedDatasets[1];

		train.show();
		test.show();

	    // Aseguramos permanencia del train en la memoria de los workers si es posible
	    train.persist();
	    
	    // Ajustamos el modelo (SVM con CV)
	    CrossValidatorModel cvModel = fitModel(train);

	    // Predicciones sobre test set
	    Dataset<Row> predictions = cvModel.transform(test).select("prediction","label");

	    // Definimos un evaluador 
	    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
			        .setMetricName("accuracy")
			        .setLabelCol("label")
			        .setPredictionCol("prediction");

	    
	    double accuracy = evaluator.evaluate(predictions);
	    System.out.println("Train samples: "+train.count());
	    System.out.println("Test samples: "+test.count());
	    System.out.println("Test Error = " + (1 - accuracy));
	    
	    ss.stop(); 
	}
}
