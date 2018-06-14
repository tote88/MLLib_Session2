package exercise_2_victor;

import java.io.BufferedReader;
import java.io.FileReader;

import java.util.HashMap;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.types.DataTypes.*;

import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;


public class exercise_2_victor {
	
	protected static Dataset<Row> getRowDataset(SparkSession ss) {
		return ss.read()
					.format("csv")
					.option("header", "true")
					.option("inferSchema", "true")
					.option("sep", ";")
					.option("nullValue", "NA")
					//.schema(schema)
					.csv("C:\\Users\\tote8\\Documents\\MBDMA\\HandsOn\\MLLib\\mllib2\\src\\main\\resources\\PacientesSim.csv");
	}

	protected static Dataset<Row> setNA (Dataset<Row> data) {
		String[] cols = {"Hemoglobina", "Creatinina", "Albumina", "Barthel", "Pfeiffer", "DiferenciaBarthel", "DiferenciaPfeiffer"};
		data = data.na().fill(-1000, cols);
		return data;
	}

	protected static Dataset<Row> discretization (Dataset<Row> data) {
		double[] splits = {Double.NEGATIVE_INFINITY, 12,42,70, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer = new Bucketizer()
				.setInputCol("DiasEstancia")
				.setOutputCol("DiasEstancia2")
				.setSplits(splits); 
		data = bucketizer.transform(data);

		double[] splitsHemoglobina = {Double.NEGATIVE_INFINITY, -999,12,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("Hemoglobina")
				.setOutputCol("Hemoglobina2")
				.setSplits(splitsHemoglobina); 
		data = bucketizer.transform(data);
		
		double[] splitsCreatinina = {Double.NEGATIVE_INFINITY, -999,1.11,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("Creatinina")
				.setOutputCol("Creatinina2")
				.setSplits(splitsCreatinina); 
		data = bucketizer.transform(data);
		
		double[] splitsAlbumina = {Double.NEGATIVE_INFINITY, -999,3.5, 5.0,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("newAlbumina")
				.setOutputCol("Albumina2")
				.setSplits(splitsAlbumina); 
		data = bucketizer.transform(data);
		
		double[] splitsBarthel = {Double.NEGATIVE_INFINITY, -999,20, 61, 91, 99,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("Barthel")
				.setOutputCol("Barthel2")
				.setSplits(splitsBarthel); 
		data = bucketizer.transform(data);
		
		double[] splitsPfeiffer = {Double.NEGATIVE_INFINITY, -999,2, 4, 8,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("Pfeiffer")
				.setOutputCol("Pfeiffer2")
				.setSplits(splitsPfeiffer); 
		data = bucketizer.transform(data);
		
		double[] splitsDifBarthel = {Double.NEGATIVE_INFINITY, -999,-20, 20,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("DiferenciaBarthel")
				.setOutputCol("DifBarthel2")
				.setSplits(splitsDifBarthel); 
		data = bucketizer.transform(data);
		
		double[] splitsDifPfeiffer = {Double.NEGATIVE_INFINITY, -999,-2, 2,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("DiferenciaPfeiffer")
				.setOutputCol("DifPfeiffer2")
				.setSplits(splitsDifPfeiffer); 
		data = bucketizer.transform(data);
		
		return data;
	}
	
	protected static Dataset<Row> dummyzation (Dataset<Row> data) {
		String[] inputCols = {"Hemoglobina2", "Creatinina2", "Albumina2", "Barthel2", "Pfeiffer2", "DifBarthel2", "DifPfeiffer2"};
		String[] outputCols = {"Hemoglobina3", "Creatinina3", "Albumina3", "Barthel3", "Pfeiffer3", "DiferenciaBarthel3", "DiferenciaPfeiffer3"};
		
		OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
				.setInputCols(inputCols)
				.setOutputCols(outputCols);
		
		data = encoder.fit(data).transform(data);
		
		String[] inputCols2 = {"IndicadorDemencia", "IndicadorConstipacion", "IndicadorSordera", "IndicadorAltVisual"};
		String[] outputCols2 = {"IndicadorDemencia3", "IndicadorConstipacion3", "IndicadorSordera3", "IndicadorAltVisual3"};
		
		//NAs
		data = data.na().fill(100,inputCols2);
		OneHotEncoderEstimator encoder2 = new OneHotEncoderEstimator()
				.setInputCols(inputCols2)
				.setOutputCols(outputCols2)
				;
		
		
		data = encoder2.fit(data).transform(data);

		return data;
	}
	
	protected static Dataset<Row> preprocessListColumns (Dataset<Row> data) {
		
		RegexTokenizer tokenizer = new RegexTokenizer()
			.setInputCol("ListaDiagnosticosPri")
			.setOutputCol("ListaDiagnosticosPri2")
			.setPattern(";");
		data = tokenizer.transform(data);
		Word2Vec word2Vec = new Word2Vec()
			.setInputCol("ListaDiagnosticosPri2")
			.setOutputCol("ListaDiagnosticosPri3");
		Word2VecModel word2VecModel = word2Vec.fit(data); 
		data = word2VecModel.transform(data);
		
		tokenizer = new RegexTokenizer()
			.setInputCol("ListaDiagnosticosSec")
			.setOutputCol("ListaDiagnosticosSec2")
			.setPattern(";");
		data = tokenizer.transform(data);
		word2Vec = new Word2Vec()
			.setInputCol("ListaDiagnosticosSec2")
			.setOutputCol("ListaDiagnosticosSec3");
		word2VecModel = word2Vec.fit(data); 
		data = word2VecModel.transform(data);
		
		tokenizer = new RegexTokenizer()
			.setInputCol("ListaProcedimientosPri")
			.setOutputCol("ListaProcedimientosPri2")
			.setPattern(";");
		data = tokenizer.transform(data);
		word2Vec = new Word2Vec()
			.setInputCol("ListaProcedimientosPri2")
			.setOutputCol("ListaProcedimientosPri3");
		word2VecModel = word2Vec.fit(data); 
		data = word2VecModel.transform(data);
		
		tokenizer = new RegexTokenizer()
			.setInputCol("ListaProcedimientosSec")
			.setOutputCol("ListaProcedimientosSec2")
			.setPattern(";");
		data = tokenizer.transform(data);
		word2Vec = new Word2Vec()
			.setInputCol("ListaProcedimientosSec2")
			.setOutputCol("ListaProcedimientosSec3");
		word2VecModel = word2Vec.fit(data); 
		data = word2VecModel.transform(data);
		
		
		tokenizer = new RegexTokenizer()
			.setInputCol("ListaCausasExternas")
			.setOutputCol("ListaCausasExternas2")
			.setPattern(";");
		data = tokenizer.transform(data);
		word2Vec = new Word2Vec()
			.setInputCol("ListaCausasExternas2")
			.setOutputCol("ListaCausasExternas3");
		word2VecModel = word2Vec.fit(data); 
		data = word2VecModel.transform(data);
		return data;
	}

	public static Dataset<Row> featurize(Dataset<Row> train) {
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"IndicadorDemencia3", "IndicadorConstipacion3", "IndicadorSordera3", "IndicadorAltVisual3",
						"Barthel3", "Pfeiffer3", "Hemoglobina3", "Creatinina3", "Albumina3",
						"DiferenciaBarthel3", "DiferenciaPfeiffer3",
						"ListaDiagnosticosPri3", "ListaDiagnosticosSec3", "ListaProcedimientosPri3", "ListaProcedimientosSec3", "ListaCausasExternas3"
				})
				.setOutputCol("features");

		Dataset<Row> featurizedDF = assembler.transform(train);
		return featurizedDF;
	}

	
	public static void main(SparkSession ss) {

		JavaSparkContext jsc = new JavaSparkContext(ss.sparkContext());
		Dataset<Row> data = getRowDataset(ss);
		data = setNA(data);

		ss.udf().register("modAlbuminaLevels", new UDF1<Double, Double>() {
			@Override
			public Double call(final Double actLevel) {
				Double newLevel;
				if (actLevel > 3.5) newLevel = 1.0;
				else newLevel = actLevel;
				return (newLevel);
			}
		}, DoubleType);
		data = data.withColumn("newAlbumina", callUDF("modAlbuminaLevels", data.col("Albumina")));
		data = discretization(data);
		data = dummyzation(data);
		data = preprocessListColumns(data);
//		data.show();

		data = featurize(data);
		Dataset<Row>[] splits= data.randomSplit(new double[] {0.3,0.7});
	    Dataset<Row> train = splits[1];
	    Dataset<Row> test = splits[0];	    
		
//	    A revisar
	 // Aseguramos permanencia del train en la memoria de los workers si es posible
//	    és necessari en aquest cas?
	    train.persist();
	    
	    // Ajustamos el modelo (SVM con CV)

		LinearSVC lsvc = new LinearSVC()
				.setMaxIter(5)
				.setLabelCol("DiasEstancia2")
				.setFeaturesCol("features");

		OneVsRest ovr = new OneVsRest()
				.setClassifier(lsvc)
				.setLabelCol("DiasEstancia2")
				.setFeaturesCol("features");

		OneVsRestModel ovrModel = ovr.fit(train);

		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(lsvc.regParam(), new double[] {1.0, 0.1, 0.01})
				.build();

		CrossValidator cv = new CrossValidator()
				.setEstimator(ovr)
				.setEvaluator(new MulticlassClassificationEvaluator()
						.setMetricName("accuracy")
						.setLabelCol("DiasEstancia2")
						.setPredictionCol("prediction"))
				.setEstimatorParamMaps(paramGrid).setNumFolds(5);

		CrossValidatorModel cvModel = cv.fit(train);

	    // Predicciones sobre test set
	    Dataset<Row> predictions = cvModel.transform(test).select("prediction","DiasEstancia2");

	    MulticlassMetrics evaluator = new MulticlassMetrics(predictions);

	    double accuracy = evaluator.accuracy();
	    System.out.println("Train samples: "+train.count());
	    System.out.println("Test samples: "+test.count());
	    System.out.println("Test Error = " + (1 - accuracy));
	    System.out.println(evaluator.confusionMatrix().toString());

	    ss.stop(); 
	}
}
