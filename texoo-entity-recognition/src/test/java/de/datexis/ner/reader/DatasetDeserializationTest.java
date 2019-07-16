package de.datexis.ner.reader;

import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.ner.MentionAnnotation;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class DatasetDeserializationTest {
  
  protected final Logger log = LoggerFactory.getLogger(getClass());
  
  @Test
  public void testReadJson() throws IOException {
    Resource data = Resource.fromJAR("datasets/kreatin.json");
    Iterator<Document> docs = ObjectSerializer.readJSONDocumentIterable(data);
    Document doc = docs.next();
    for(Annotation ann : doc.getAnnotations()) {
      if(ann instanceof MentionAnnotation) Assert.assertEquals(Annotation.Source.PRED, ann.getSource());
      if(ann instanceof PassageAnnotation) Assert.assertEquals(Annotation.Source.UNK, ann.getSource());
    }
    Assert.assertEquals(18, doc.countAnnotations());
    Assert.assertEquals(6, doc.countAnnotations(PassageAnnotation.class));
    Assert.assertEquals(12, doc.countAnnotations(MentionAnnotation.class));
    Assert.assertEquals(6, doc.countAnnotations(Annotation.Source.UNK));
    Assert.assertEquals(12, doc.countAnnotations(Annotation.Source.PRED));
  }
  
}
