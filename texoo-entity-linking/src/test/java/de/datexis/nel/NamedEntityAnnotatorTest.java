package de.datexis.nel;

import de.datexis.ner.MentionAnnotator;
import de.datexis.common.Resource;
import de.datexis.encoder.AbstractEncoder;
import de.datexis.encoder.impl.LetterNGramEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.index.ArticleIndex;
import de.datexis.ner.MentionAnnotation;
import de.datexis.index.ArticleIndexFactory;
import de.datexis.ner.GenericMentionAnnotator;
import java.io.IOException;
import org.junit.Test;
import org.junit.Before;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class NamedEntityAnnotatorTest {
  
  final String text = "Berlin is the capital and the largest city of Germany as well as one of its constituent 16 states. With a population of approximately 3.5 million, Berlin is the second most populous city proper and the seventh most populous urban area in the European Union.";
  
  MentionAnnotator recognize;
  ArticleIndex search;
  AbstractEncoder disambiguate;
  
  NamedEntityAnnotator ann;
  
  @Before
  public void init() throws IOException {
    recognize = GenericMentionAnnotator.create();
    Resource index = Resource.fromJAR("models/Articles_en_Wikidata_250_20170828.txt");
    search = ArticleIndexFactory.loadWikiDataIndex(index);
    Resource trigram = Resource.fromJAR("models/MentionAnnotator_en_NER-GENERIC_WikiNER+tri_20170309/trigram.tsv.gz");
    disambiguate = new LetterNGramEncoder(3);
    disambiguate.loadModel(trigram);
    ann = new NamedEntityAnnotator(recognize, search, disambiguate);
  }
  
  @Test
  public void test() {
    
    Document doc = ann.annotate(text);
    // Retrieve the first Document and print
    System.out.println(String.format("Document [%s]: \"%s\"", doc.getLanguage(), doc.getText()));
    // Retrieve all Annotations and print
    doc.streamAnnotations(Annotation.Source.PRED, MentionAnnotation.class).forEach(ann -> {
      System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
    });
    doc.streamAnnotations(Annotation.Source.PRED, NamedEntityAnnotation.class).forEach(ann -> {
      System.out.println(String.format("++ %s [%s]\t%s", ann.getText(), ann.getRefId(), ann.getConfidence()));
    });
  }
  
}
