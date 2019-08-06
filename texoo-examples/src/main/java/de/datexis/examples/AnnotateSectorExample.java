package de.datexis.examples;

import de.datexis.annotator.AnnotatorFactory;
import de.datexis.common.AnnotationHelpers;
import de.datexis.common.DialogHelpers;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.sector.SectorAnnotator;
import de.datexis.sector.encoder.ClassEncoder;
import de.datexis.sector.encoder.HeadingEncoder;
import de.datexis.sector.model.SectionAnnotation;
import de.datexis.sector.reader.WikiSectionReader;
import de.datexis.sector.tagger.SectorEncoder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;

/**
 * Run experiments on a pre-trained SECTOR model
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AnnotateSectorExample {

  protected final static Logger log = LoggerFactory.getLogger(AnnotateSectorExample.class);

  /**
   * This example requires a SECTOR model stored on disk.
   * @param args : [0] full path to WikiSection dataset, e.g. /home/texoo/WikiSection_v2.0/wikisection_en_disease_validation.json
   *                [1] full path to Sector Model directory, e.g. /home/texoo/SectorAnnotator_WikiSection_en_disease/
   */
  public static void main(String[] args) throws IOException {
    String modelDir = args.length > 0 ? args[0] : 
        DialogHelpers.askForDirectory("Please choose a Sector model directory");
    String datasetFile = args.length > 1 ? args[1] : 
        DialogHelpers.askForFile("Please choose a WikiSection dataset", new FileNameExtensionFilter("WikiSection dataset", "json.gz", "json"));
    new AnnotateSectorExample().evalClassifier(modelDir, datasetFile);
  }
  
  public AnnotateSectorExample() {}
  
  public void evalClassifier(String modelDir, String datasetFile) throws IOException {
    
    Resource datasetPath = Resource.fromDirectory(datasetFile);
    Resource modelPath = Resource.fromDirectory(modelDir);
    
    Nd4j.create(1);
    
    // --- load model ----------------------------------------------------------
    SectorAnnotator sector = (SectorAnnotator) AnnotatorFactory.loadAnnotator(modelPath);
    HeadingEncoder headings = ((HeadingEncoder)sector.getComponent("HL"));
    ClassEncoder labels = ((ClassEncoder)sector.getComponent("CLS"));
    
    // --- load data -----------------------------------------------------------
    Dataset validation = WikiSectionReader.readDatasetFromJSON(datasetPath).getSplit(0, 16);
    
    // --- annotate documents --------------------------------------------------
    sector.annotate(validation.getDocuments(), SectorAnnotator.SegmentationMethod.MAX);
    sector.evaluateModel(validation);
    
    // --- print some examples -------------------------------------------------
    printSentences(validation.getDocument(0).get(), headings, labels);
  }
  
  private static void printSentences(Document doc, HeadingEncoder headings, ClassEncoder labels) {
    System.out.println("GOLD HEADING\tPRED HEADINGS\tPRED LABEL\tEMBEDDING\tTEXT");
    for(Sentence s : doc.getSentences()) {
      // These are the Annotations for this Sentence:
      SectionAnnotation gold = AnnotationHelpers.getAnnotationsForSpan(doc, Annotation.Source.GOLD, SectionAnnotation.class, s).stream().findFirst().get();
      SectionAnnotation pred = AnnotationHelpers.getAnnotationsForSpan(doc, Annotation.Source.PRED, SectionAnnotation.class, s).stream().findFirst().get();
      // This is the topic embedding:
      INDArray embedding = s.getVector(SectorEncoder.class);
      // This is the headline prediction:
      INDArray heading = s.getVector(HeadingEncoder.class);
      // This is the class label prediction:
      INDArray label = s.getVector(ClassEncoder.class);
      // get the list of predicted labels with confidence per sentence as follows:
      Collection<Map.Entry<String,Double>> predictions = labels.getNearestNeighbourEntries(label, 5);
      System.out.println(
          gold.getSectionHeading() + "\t" + 
          headings.getNearestNeighbours(heading, 5).toString() + "\t" + 
          pred.getSectionLabel() + "\t" + 
          embedding.sumNumber().doubleValue() + "\t" + 
          s.getText()
      );
    }
    
  }
    
}
