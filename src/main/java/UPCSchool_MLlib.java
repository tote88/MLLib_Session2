

import org.apache.spark.sql.SparkSession;
import exercise_pacientes.exercise_pacientes;

 

public class UPCSchool_MLlib {
	/*
		Added Winutils because it did not work properly on windows (caused exception with ls function on wholetextfiles
	 */
	static String HADOOP_COMMON_PATH = "C:\\Users\\tote8\\Documents\\MBDMA\\HandsOn\\MLLib\\Spam_Detection_empty\\Spam_Detection_empty\\src\\main\\resources\\winutils";
	public static void main(String[] args) throws Exception {
		System.setProperty("hadoop.home.dir", HADOOP_COMMON_PATH);
		SparkSession spark = SparkSession.builder().master("local[*]")
				.appName("spamDetection")
				.getOrCreate();
		exercise_pacientes.main(spark);
		spark.close();
	}

}
