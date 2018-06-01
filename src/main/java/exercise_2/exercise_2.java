package exercise_2;


import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.types.DataTypes.*;


public class exercise_2 {

	public static void pacientesModel(SparkSession ss) {

		JavaSparkContext jsc = new JavaSparkContext(ss.sparkContext());

		Dataset<Row> pacientes = getRowDataset(ss);

		Dataset<Row> splitted = splitDiasEstancia(pacientes);

		Dataset<Row> withoutNA = imputeValueToNA(splitted);

		Dataset<Row> categorised = categorise(withoutNA);

		Dataset<Row> barthel = categoriseBarthel(categorised);

		Dataset<Row> pfeiffer = categorisePfeiffer(barthel);

		Dataset<Row> DifBarthel = categoriseDifBarthel(pfeiffer);

		Dataset<Row> DifPfeiffer = categoriseDifPfeiffer(DifBarthel);

		OneHotEncoderEstimator oneHotEncoderEstimator = new OneHotEncoderEstimator()
				.setInputCols(new String[]{"BarthelD", "PfeifferD", "DifBarthelD", "DifPfeifferD", "HemoglobinaD", "AlbuminaD", "CreatininaD",})
				.setOutputCols(new String[]{"BarthelDummy", "PfeifferDummy", "DifBarthelDummy", "DifPfeifferDummy", "HemoglobinaDummy", "AlbuminaDummy", "CreatininaDummy"});


		Dataset<Row> transform = oneHotEncoderEstimator.fit(DifPfeiffer).transform(DifPfeiffer);

		transform.show();

	    ss.stop(); 
	}

	private static Dataset<Row> categoriseDifPfeiffer(Dataset<Row> in) {
		double[] splits =
				{Double.NEGATIVE_INFINITY, -999, -2, 2, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer = new Bucketizer()
				.setInputCol("DiferenciaPfeiffer") //To be removed (?)
				.setOutputCol("DifPfeifferD")
				.setSplits(splits);

		return bucketizer.transform(in);
	}

	private static Dataset<Row> categoriseDifBarthel(Dataset<Row> in) {
		double[] splits =
				{Double.NEGATIVE_INFINITY, -999, -20, 20, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer = new Bucketizer()
				.setInputCol("DiferenciaBarthel") //To be removed (?)
				.setOutputCol("DifBarthelD")
				.setSplits(splits);

		return bucketizer.transform(in);
	}

	private static Dataset<Row> categorisePfeiffer(Dataset<Row> barthel) {
		double[] splits =
				{Double.NEGATIVE_INFINITY, -999, 2, 4, 8, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer = new Bucketizer()
				.setInputCol("Pfeiffer") //To be removed (?)
				.setOutputCol("PfeifferD")
				.setSplits(splits);

		return bucketizer.transform(barthel);
	}

	private static Dataset<Row> categoriseBarthel(Dataset<Row> categorised) {
		double[] splits =
				{Double.NEGATIVE_INFINITY, -999, 20, 61, 91, 99, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer = new Bucketizer()
				.setInputCol("Barthel") //To be removed (?)
				.setOutputCol("BarthelD")
				.setSplits(splits);

		return bucketizer.transform(categorised);
	}

	private static Dataset<Row> categorise(Dataset<Row> withoutNA) {
		// Hemoglobina
		double[] splitHemoglobina =
				{Double.NEGATIVE_INFINITY, -999, 12.0, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer = new Bucketizer()
				.setInputCol("Hemoglobina") //To be removed (?)
				.setOutputCol("HemoglobinaD")
				.setSplits(splitHemoglobina);

		Dataset<Row> step1 = bucketizer.transform(withoutNA);

		double[] splitsCreatinina =
				{Double.NEGATIVE_INFINITY, -999, 1.11, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer2 = new Bucketizer()
				.setInputCol("Creatinina") //To be removed (?)
				.setOutputCol("CreatininaD")
				.setSplits(splitsCreatinina);

		Dataset<Row> step2 = bucketizer2.transform(step1);

		double[] splitsAlbumina =
				{Double.NEGATIVE_INFINITY, -999, 3.5, 5.0, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer3 = new Bucketizer()
				.setInputCol("Albumina") //To be removed (?)
				.setOutputCol("AlbuminaD")
				.setSplits(splitsAlbumina);

		Dataset<Row> step3 = bucketizer3.transform(step2);

		return step3;
	}

	private static Dataset<Row> imputeValueToNA(Dataset<Row> splitted) {
//		String[] colImputar = {"Hemoglobina", "Creatinina", "Albumina", "Barthel", "Pfeiffer", "DifBarthel", "DifPfeiffer"};
		String[] colImputar = {"Hemoglobina", "Creatinina", "Albumina", "Barthel", "Pfeiffer", "DifBarthel", "DifPfeiffer", "IndicadorDemencia", "IndicadorConstipacion", "IndicadorSordera", "IndicadorAltVisual", "DiferenciaBarthel", "DiferenciaPfeiffer"};
		return splitted.na().fill(-1000, colImputar);
	}

	private static Dataset<Row> splitDiasEstancia(Dataset<Row> pacientes) {
		double[] splits =
				{Double.NEGATIVE_INFINITY, 12, 42, 70, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer = new Bucketizer()
				.setInputCol("DiasEstancia") //To be removed (?)
				.setOutputCol("DiasEstanciaD")
				.setSplits(splits);

		return bucketizer.transform(pacientes);
	}

	private static Dataset<Row> getRowDataset(SparkSession ss) {
		StructType schema = new StructType()
				.add("Id", "integer")
				.add("IndicadorDemencia", "integer")
				.add("IndicadorConstipacion", "integer")
				.add("IndicadorSordera", "integer")
				.add("IndicadorAltVisual", "integer")
				.add("Barthel", "integer")
				.add("Pfeiffer", "integer")
				.add("DiferenciaBarthel", "integer")
				.add("DiferenciaPfeiffer", "integer")
				.add("Hemoglobina", "float")
				.add("Creatinina", "float")
				.add("Albumina", "float")
				.add("ListaDiagnosticosPri", "string")
				.add("ListaDiagnosticosSec", "string")
				.add("ListaProcedimientosPri", "string")
				.add("ListaProcedimientosSec", "string")
				.add("ListaCausasExternas", "string")
				.add("Reingreso", "integer")
				.add("DiasEstancia", "integer");

		return ss.read()
					.format("csv")
					.option("header", "true")
					//.option("inferSchema", "true")
					.option("sep", ";")
					.option("nullValue", "NA")
					.schema(schema)
					.csv("C:\\Users\\tote8\\Documents\\MBDMA\\HandsOn\\MLLib\\mllib2\\src\\main\\resources\\PacientesSim.csv");
	}
}
