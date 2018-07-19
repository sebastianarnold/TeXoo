package de.datexis.examples;

import de.datexis.models.ner.MentionAnnotator;
import de.datexis.common.*;
import de.datexis.encoder.impl.*;
import de.datexis.models.ner.eval.MentionAnnotatorEval;
import de.datexis.model.*;
import de.datexis.models.ner.MentionAnnotation;
import de.datexis.model.tag.BIOESTag;
import de.datexis.models.ner.eval.MentionAnnotatorEvaluation;
import de.datexis.preprocess.DocumentFactory;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Example: how to train a new Model using an in-memory Dataset
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class TrainEntityRecognition {

  protected final static Logger log = LoggerFactory.getLogger(TrainEntityRecognition.class);

  public static void main(String[] args) {
    
    // --- preprocessing -------------------------------------------------------
    // parse texts into Documents
    Document trainDoc = DocumentFactory.fromText(
        "TeXoo ist ein Framework der Gruppe DATEXIS an der Beuth Hochschule für Technik Berlin. " +
        "Die Gruppe Datexis steht für Datenbanken und Textbasierte Informationssysteme."
    );
    Document testDoc = DocumentFactory.fromText(
        "Was für ein Framework ist TeXoo? Wurde es von der Gruppe DATEXIS hergestellt?"
    );
    
    // add some GOLD annotations
    trainDoc.addAnnotation(new MentionAnnotation(Annotation.Source.GOLD, Arrays.asList(trainDoc.getToken(0).get()))); // TeXoo
    trainDoc.addAnnotation(new MentionAnnotation(Annotation.Source.GOLD, Arrays.asList(trainDoc.getToken(6).get()))); // DATEXIS
    trainDoc.addAnnotation(new MentionAnnotation(Annotation.Source.GOLD, Arrays.asList(trainDoc.getToken(9).get(), trainDoc.getToken(10).get(), trainDoc.getToken(11).get(), trainDoc.getToken(12).get(), trainDoc.getToken(13).get())));
    trainDoc.addAnnotation(new MentionAnnotation(Annotation.Source.GOLD, Arrays.asList(trainDoc.getToken(17).get()))); // Datexis
    testDoc.addAnnotation(new MentionAnnotation(Annotation.Source.GOLD, Arrays.asList(testDoc.getToken(5).get()))); // TeXoo
    testDoc.addAnnotation(new MentionAnnotation(Annotation.Source.GOLD, Arrays.asList(testDoc.getToken(12).get()))); // DATEXIS
    
    // add Tags to Tokens based on Annotation
    MentionAnnotation.createTagsFromAnnotations(trainDoc, Annotation.Source.GOLD, BIOESTag.class);
    MentionAnnotation.createTagsFromAnnotations(testDoc, Annotation.Source.GOLD, BIOESTag.class);
    
    // add this Document into Datasets - for a real example we need around 4.000 training sentences
    Dataset train = new Dataset("training data", Arrays.asList(trainDoc));
    Dataset test  = new Dataset("test data", Arrays.asList(testDoc));
    
    // --- configuration -------------------------------------------------------
    // configure and build Annotator and Encoders
    MentionAnnotator ann = new MentionAnnotator.Builder()
        .withEncoders("tri", new PositionEncoder(), new SurfaceEncoder(), new TrigramEncoder())
        .enableTrainingUI(false)
        .withTrainingParams(0.01, 16, 50)
        .withWorkspaceParams(1)
        .pretrain(train)
        .build();
    
    // --- training ------------------------------------------------------------
    // create the Tagger iterator and finally train the model
    ann.trainModel(train);
    
    // --- test ----------------------------------------------------------------
    // test the model
    ann.annotate(test);
    MentionAnnotatorEvaluation eval = new MentionAnnotatorEvaluation(ann.getTagger().getName(), Annotation.Match.STRONG);
    eval.calculateScores(test);
    ann.getTagger().appendTestLog(eval.printAnnotationStats());
    // Retrieve all Annotations and print predictions
    for(MentionAnnotation a : testDoc.getAnnotations(Annotation.Source.PRED, MentionAnnotation.class)) {
        System.out.println(String.format("-- %s [%s]\t%s", a.getText(), a.getType(), a.getConfidence()));
        System.out.println(ObjectSerializer.getJSON(a));
    }
    
    // --- save ----------------------------------------------------------------
    // save models to a new path, you can configure this in texoo.properties
    Resource outputPath = ExportHandler.createExportPath("ExampleNER");
    System.out.println("saving model to path: " + outputPath);
    ann.writeModel(outputPath, "annotator"); // save the complete model as XML
    ann.writeTrainLog(outputPath); // write log data
    ann.writeTestLog(outputPath);
    
  }
  
}
