package de.datexis.models.sector.model;

import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import java.io.IOException;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class SectionAnnotatorTest {
  
  public SectionAnnotatorTest() {
  }
  
  @Test
  public void testAnnotationEvaluation() throws IOException {
    Resource result = Resource.fromJAR("results/chediak_higashi_syndrome.json");
    Document doc = ObjectSerializer.readFromJSON(result, Document.class);

    System.out.println("GOLD");
    for(SectionAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, SectionAnnotation.class)) {
      System.out.println(ann.getBegin() + "\t" + ann.sectionLabel);
    }

    System.out.println("PRED");
    for(SectionAnnotation ann : doc.getAnnotations(Annotation.Source.PRED, SectionAnnotation.class)) {
      System.out.println(ann.getBegin() + "\t" + ann.sectionLabel + "\t" + ann.getConfidence());
    }
    
  }
  
}
