

import exercise_2.exercise_2;
import org.apache.spark.sql.SparkSession;
import exercise_1.exercise_1;
import exercise_2_victor.exercise_2_victor;

 

public class UPCSchool_MLlib {
	/*
		Added Winutils because it did not work properly on windows (Exception with ls function on wholetextfiles
	 */
	static String HADOOP_COMMON_PATH = "C:\\Users\\tote8\\Documents\\MBDMA\\HandsOn\\MLLib\\Spam_Detection_empty\\Spam_Detection_empty\\src\\main\\resources\\winutils";
	public static void main(String[] args) throws Exception {
		System.setProperty("hadoop.home.dir", HADOOP_COMMON_PATH);
		SparkSession spark = SparkSession.builder().master("local[*]")
				.appName("spamDetection")
				.getOrCreate();
		//exercise_2.pacientesModel(spark);
		exercise_2_victor.main(spark);
		spark.close();
	}

}
