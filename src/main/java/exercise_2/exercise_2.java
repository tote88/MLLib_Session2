package exercise_2;


import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.Bucketizer;
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

		categorised.show();

	    ss.stop(); 
	}

	private static Dataset<Row> categorise(Dataset<Row> withoutNA) {
		// Hemoglobina
		double[] splits =
				{Double.NEGATIVE_INFINITY, -1000, 12.0, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer = new Bucketizer()
				.setInputCol("Hemoglobina") //To be removed (?)
				.setOutputCol("HemoglobinaD")
				.setSplits(splits);

		Dataset<Row> step1 = bucketizer.transform(withoutNA);

		double[] splits2 =
				{Double.NEGATIVE_INFINITY, -1000, 1.11, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer2 = new Bucketizer()
				.setInputCol("Creatinina") //To be removed (?)
				.setOutputCol("CreatininaD")
				.setSplits(splits2);

		Dataset<Row> step2 = bucketizer2.transform(step1);

		double[] splits3 =
				{Double.NEGATIVE_INFINITY, -1000, 3.5, 5.0, Double.POSITIVE_INFINITY};
		Bucketizer bucketizer3 = new Bucketizer()
				.setInputCol("Albumina") //To be removed (?)
				.setOutputCol("AlbuminaD")
				.setSplits(splits3);

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
