package de.datexis.cdv.index;

import de.datexis.cdv.preprocess.AspectPreprocessor;
import de.datexis.cdv.retrieval.EntityAspectQueryAnnotation;
import de.datexis.encoder.IEncoder;
import de.datexis.model.Dataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AspectIndexBuilder {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  protected static String[][] getAspectHeadingAssignments(String datasetName) {
  
    // select representative headings for each class
    String[][] aspectAssignments = null;
    if(datasetName.equals("MedQuAD")) aspectAssignments = new String[][]{
      {"information", "description"},
      {"exams_and_tests", "diagnosis"},
      {"treatment", "treatment"},
      {"symptoms", "symptoms and presentation"},
      {"prevention", "prevention"},
      {"stages", "staging"},
      {"genetic_changes", "genetic mechanism"},
      {"causes", "causes and mechanism"},
      {"considerations", "management considerations"},
      {"susceptibility", "risk factors"},
      {"research", "research directions and trials"},
      {"frequency", "frequency and epidemiology"},
      {"complications", "complications"},
      {"inheritance", "genetic inheritance"},
      {"outlook", "prognosis"},
    };
  
    if(datasetName.equals("WikiSection")) aspectAssignments = new String[][]{
      {"information", "description"},
      {"cause", "causes and pathogenesis"},
      {"classification", "classification and types"},
      {"complication", "complications"},
      {"culture", "society and culture"},
      {"diagnosis", "diagnosis"},
      {"epidemiology", "epidemiology"},
      {"etymology", "terminology and etymology"},
      {"fauna", "animals"},
      {"genetics", "genetic mechanism"},
      {"geography", "geography"}, // where
      {"history", "history"}, // where
      {"infection", "infection and transmission"},
      {"management", "management"},
      {"mechanism", "mechanism"},
      {"medication", "medication and drugs"}, // where
      {"pathology", "pathology"},
      {"pathophysiology", "pathophysiology"},
      {"prevention", "prevention"},
      {"prognosis", "prognosis"},
      {"research", "research directions and trials"},
      {"risk", "risk factors"},
      {"screening", "screening tests"},
      {"surgery", "surgical procedures"}, // where
      {"symptom", "signs and symptoms"},
      {"tomography", "tomography"},
      {"treatment", "treatment"},
    };
    if(datasetName.equals("HealthQA")) aspectAssignments = new String[][]{
      {"information", "description"},
      {"diagnosis", "diagnosis"},
      {"treatment", "treatment"},
      {"risk_factors", "risk factors"},
      {"complications", "complications"},
      {"symptoms", "signs and symptoms"},
      {"causes", "causes and pathogenesis"},
      {"surgery", "surgical procedures"},
      {"severity", "severity"},
      {"impact", "impact on life"},
      {"prognosis", "prognosis"},
      {"prevalence", "prevalence and epidemiology"},
      {"tests", "tests and screening"},
      {"prevention", "prevention"},
      {"recovery", "recovery and healing"},
      {"types", "classification and types"},
      {"medication", "medication and drugs"},
      {"infection", "infection and transmission"},
      {"management", "management considerations"},
      {"inheritance", "genetic inheritance"},
      {"mechanism", "mechanism and staging"},
    };
  
    return aspectAssignments;
  }
  
  public static AspectIndex buildAspectIndex(IEncoder encoder, String datasetName) {
    String[][] aspectAssignments = getAspectHeadingAssignments(datasetName);
    AspectIndex labels = new AspectIndex(encoder);
    labels.buildKeyIndex(Arrays.stream(aspectAssignments).map(arr -> arr[0]).collect(Collectors.toList()), false);
    Map<String, String> examples = new HashMap<>();
    Arrays.stream(aspectAssignments).forEach(arr -> Arrays.stream(arr).skip(1).forEach( h -> {
      examples.put(arr[0], h);
    }));
    labels.encodeAndBuildVectorIndex(examples, false); // lookup headings that were already encoded
    return labels;
  }
  
  public static void appendQueryAspectHeadings(Dataset corpus, String datasetName) {
    AspectPreprocessor preprocessor = new AspectPreprocessor();
    String[][] aspectAssignments = getAspectHeadingAssignments(datasetName);
    Map<String, String> examples = new HashMap<>();
    Arrays.stream(aspectAssignments).forEach(arr -> Arrays.stream(arr).skip(1).forEach( h -> {
      examples.put(arr[0], h);
    }));
    corpus.getQueries().forEach(q -> {
      EntityAspectQueryAnnotation ann = q.getAnnotation(EntityAspectQueryAnnotation.class);
      ann.setAspectHeading(ann.getAspect().equals("information") ? "information" :
        examples.get(preprocessor.preProcess(ann.getAspect())));
    });
  }
  
}
