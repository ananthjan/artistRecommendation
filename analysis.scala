import java.io.File
import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.DoubleType
import math._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}


// Load the dataset from hdfs

val df = spark.read
        .format("com.databricks.spark.csv")
        .option("header", "true") 
        .load("SongCSV.txt")
df.describe()
df.printSchema()

df.select("ArtistID", "SongID").groupBy("ArtistID").count().show()


val newdf = df.withColumn("SongNumber", df("SongNumber").cast(IntegerType)).withColumn("AlbumID", df("AlbumID").cast(IntegerType)).withColumn("ArtistLatitude", df("ArtistLatitude").cast(DoubleType)).withColumn("ArtistLongitude", df("ArtistLongitude").cast(DoubleType)).withColumn("Danceability", df("Danceability").cast(IntegerType)).withColumn("Duration", df("Duration").cast(DoubleType)).withColumn("KeySignature", df("KeySignature").cast(IntegerType)).withColumn("KeySignatureConfidence", df("KeySignatureConfidence").cast(DoubleType)).withColumn("Popularity", df("Popularity").cast(DoubleType)).withColumn("Tempo", df("Tempo").cast(DoubleType)).withColumn("TimeSignature", df("TimeSignature").cast(IntegerType)).withColumn("TimeSignatureConfidence", df("TimeSignatureConfidence").cast(DoubleType)).withColumn("Year", df("Year").cast(IntegerType)).withColumn("EndofFadein", df("EndofFadein").cast(DoubleType)).withColumnRenamed("Title","songTitle")

val Projected = newdf.select("songTitle", "ArtistLatitude", "ArtistLocation", "ArtistLongitude", "ArtistName", "Popularity") 

//"Filtered based popularity and non-null Latitude and Longitude"
val Filtered = Projected.filter($"Popularity" >= 0.5).filter($"ArtistLatitude".isNotNull).filter($"ArtistLongitude".isNotNull)

Filtered.registerTempTable("Filtered")

val Filtered2 = sqlContext.sql("SELECT songTitle as song2Title, ArtistLatitude as artist2Lat, ArtistLongitude as artist2Long from Filtered")

//"Produce all pairs of different songs and calculate distance between localizations of their artists"
val Crossed = Filtered2.join(Filtered)

Crossed.registerTempTable("Crossed")



val Different = sqlContext.sql("SELECT * from Crossed where songTitle != song2Title")

Different.registerTempTable("Different")

def haversine(lat1:Double, lon1:Double, lat2:Double, lon2:Double) : Double ={
      val R = 6372.8
      val dLat=(lat2 - lat1).toRadians
      val dLon=(lon2 - lon1).toRadians
 
      val a = pow(sin(dLat/2),2) + pow(sin(dLon/2),2) * cos(lat1.toRadians) * cos(lat2.toRadians)
      val c = 2 * asin(sqrt(a))
      val dist_in_km = R * c	
      val dist_in_miles = dist_in_km * 0.62137
      return dist_in_miles
   }

sqlContext.udf.register("haversine",haversine _)

val Distanced = sqlContext.sql("select songTitle, song2Title, ArtistLatitude, ArtistLongitude, ArtistName, Popularity,haversine(ArtistLatitude, ArtistLongitude, artist2Lat, artist2Long) AS distance from Different")

Distanced.registerTempTable("Distanced")

//"For each song, calculate average distance between its artists and all other artists"
val AvgDistanced = sqlContext.sql("select songTitle, ArtistLatitude, ArtistLongitude, ArtistName, Popularity, AVG(distance) AS distanceAvg from Distanced GROUP BY songTitle, ArtistLatitude, ArtistLongitude, ArtistName, Popularity")

AvgDistanced.registerTempTable("AvgDistanced")

AvgDistanced.show()


//"Find the most popular song for a given artist location"
//val poptemp = sqlContext.sql("SELECT MAX(Popularity) as maxpop,ArtistLatitude, ArtistLongitude FROM AvgDistanced GROUP BY ArtistLatitude, ArtistLongitude")

//poptemp.registerTempTable("poptemp")

//val Popular = sqlContext.sql("Select poptemp.maxpop, poptemp.ArtistLatitude, poptemp.ArtistLongitude, songTitle, ArtistName, distanceAvg from poptemp, AvgDistanced WHERE poptemp.maxpop = AvgDistanced.Popularity AND poptemp.ArtistLatitude = AvgDistanced.ArtistLatitude AND poptemp.ArtistLongitude = AvgDistanced.ArtistLongitude ")

//Popular.registerTempTable("Popular")

//Popular.show()




val Art_pop = sqlContext.sql("Select ArtistName, ArtistLatitude, ArtistLongitude, AVG(Popularity) from AvgDistanced GROUP BY ArtistName, ArtistLatitude, ArtistLongitude")

Art_pop.registerTempTable("Art_pop")

Art_pop.show()


val startlat = 40.65507
val startlng = -73.94888

//startlat

//val Nearby_artist = sqlContext.sql("SELECT ArtistLatitude, ArtistLongitude, SQRT(POW(69.1 * (ArtistLatitude - 'startlat'), 2) + POW(69.1 * ('startlng' - ArtistLongitude) * COS(ArtistLatitude / 57.3), 2)) AS distance FROM Art_pop HAVING distance < 50 ORDER BY distance")

val Nearby_artist = sqlContext.sql("SELECT ArtistName, ArtistLatitude, ArtistLongitude, SQRT(POW(69.1 * (ArtistLatitude - 40.65507), 2) + POW(69.1 * (-73.94888 - ArtistLongitude) * COS(ArtistLatitude / 57.3), 2)) AS distance FROM Art_pop HAVING distance < 25 ORDER BY distance")

Nearby_artist.show()

Nearby_artist.describe()


Nearby_artist.rdd.saveAsTextFile("nearby_Artists")