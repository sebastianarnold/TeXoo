-- MySQL dump 10.16  Distrib 10.1.26-MariaDB, for debian-linux-gnu (x86_64)
--
-- Host: 10.64.48.13    Database: enwiki
-- ------------------------------------------------------
-- Server version	10.1.33-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `redirect`
--

DROP TABLE IF EXISTS `redirect`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `redirect` (
  `rd_from` int(8) unsigned NOT NULL DEFAULT '0',
  `rd_namespace` int(11) NOT NULL DEFAULT '0',
  `rd_title` varbinary(255) NOT NULL DEFAULT '',
  `rd_interwiki` varbinary(32) DEFAULT NULL,
  `rd_fragment` varbinary(255) DEFAULT NULL,
  PRIMARY KEY (`rd_from`),
  KEY `rd_ns_title` (`rd_namespace`,`rd_title`,`rd_from`)
) ENGINE=InnoDB DEFAULT CHARSET=binary;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `redirect`
--

/*!40000 ALTER TABLE `redirect` DISABLE KEYS */;
INSERT INTO `redirect` VALUES (10,0,'Computer_accessibility','',''),(13,0,'History_of_Afghanistan','',''),(14,0,'Geography_of_Afghanistan','',''),(15,0,'Demographics_of_Afghanistan','',''),(18,0,'Communications_in_Afghanistan','',''),(19,0,'Transport_in_Afghanistan','',''),(20,0,'Afghan_Armed_Forces','',''),(21,0,'Foreign_relations_of_Afghanistan','',''),(23,0,'Assistive_technology','',''),(24,0,'Amoeba','',''),(27,0,'History_of_Albania','',''),(29,0,'Demographics_of_Albania','',''),(30,0,'As_We_May_Think','',''),(35,0,'Politics_of_Albania','',''),(36,0,'Economy_of_Albania','',''),(40,0,'Afroasiatic_languages','',''),(42,0,'Constructed_language','',''),(46,0,'Abacus','',''),(47,0,'Abalone','',''),(48,0,'Abbadid_dynasty','',''),(49,0,'Abbess','',''),(50,0,'Abbeville','',''),(51,0,'Abbey','',''),(52,0,'Abbot','',''),(53,0,'Abbreviation','',''),(54,0,'Atlas_Shrugged','',''),(56,0,'Constructed_language','',''),(58,0,'List_of_Atlas_Shrugged_characters','',''),(59,0,'Atlas_Shrugged','',''),(60,0,'Atlas_Shrugged','',''),(241,0,'African_Americans','',''),(242,0,'Adolf_Hitler','',''),(247,0,'Abecedarian','',''),(248,0,'Cain_and_Abel','',''),(249,0,'Abensberg','',''),(251,0,'Aberdeen,_South_Dakota','',''),(254,0,'Arthur_Koestler','',''),(255,0,'Ayn_Rand','',''),(256,0,'Alexander_the_Great','',''),(258,0,'Anchorage,_Alaska','',''),(259,0,'Logical_form','',''),(260,0,'Existence_of_God','',''),(263,0,'Anarchy','',''),(264,0,'ASCII_art','',''),(269,0,'Academy_Awards','',''),(270,0,'Academy_Award_for_Best_Picture','',''),(271,0,'Austrian_German','',''),(272,0,'Ivory_tower','',''),(274,0,'Axiom_of_choice','',''),(276,0,'American_football','',''),(278,0,'United_States','',''),(279,0,'Anna_Kournikova','',''),(280,0,'Andorra','',''),(287,0,'Austroasiatic_languages','',''),(289,0,'Lists_of_actors','',''),(291,0,'Anarcho-capitalism','',''),(293,0,'Anarcho-capitalism','',''),(296,0,'Lists_of_actors','',''),(299,0,'An_American_in_Paris','',''),(301,0,'Automorphism','',''),(302,0,'Action_film','',''),(304,0,'Africa','',''),(306,0,'Statistics','',''),(325,0,'Action_film','',''),(338,0,'Auto_racing','',''),(347,0,'Demographics_of_Algeria','',''),(353,0,'Foreign_relations_of_Algeria','',''),(369,0,'Atlas_Shrugged','',''),(583,0,'Amoeba','',''),(589,0,'Ashmore_and_Cartier_Islands','',''),(596,0,'Constructed_language','',''),(598,0,'Afroasiatic_languages','',''),(609,0,'Foreign_relations_of_Andorra','',''),(617,0,'Al_Gore','',''),(618,0,'An_Enquiry_Concerning_Human_Understanding','',''),(622,0,'Al_Gore','',''),(626,0,'Auteur','',''),(629,0,'Abstract_algebra','',''),(635,0,'Analysis_of_variance','',''),(644,0,'Arithmetic_logic_unit','',''),(648,0,'Actor','',''),(654,0,'Computer_accessibility','','');
INSERT INTO `redirect` VALUES (654,0,'Computer_accessibility','',''),(179642,0,'Computer_accessibility','',''),(1641854,0,'Computer_accessibility','',''),(56315086,0,'Computer_accessibility','','Open Accessibility Framework');


/*!40000 ALTER TABLE `redirect` ENABLE KEYS */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-01-01 10:42:48
