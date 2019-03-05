package de.datexis.sector.reader;

import com.google.common.collect.Lists;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.sector.model.SectionAnnotation;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThat;

public class ChoiDatasetReaderTest {

  private static final Logger log = LoggerFactory.getLogger(ChoiDatasetReaderTest.class);
  public static final String DATASET_NAME = "test";
  public static final String SUFFIX = ".tmp";
  protected Resource CHOI_DATASET_EXAMPLE;

  @Before
  public void setUp() throws Exception {
    CHOI_DATASET_EXAMPLE = Resource.fromJAR("testdata/choi-1-3-5-0.ref");
  }

  @Test
  public void readDatasetFromPath() throws Exception {
    Dataset test = new ChoiDatasetReader().read(CHOI_DATASET_EXAMPLE);
    assertNotNull(test);
  }  
  
  @Test
  public void shouldReadCorrectNumberOfDocumentsSentencesAndAnnotationsFromRealWorldExample() throws Exception {
    Dataset test = new ChoiDatasetReader().read(CHOI_DATASET_EXAMPLE);
    assertThat(test.countDocuments(), is(equalTo(1)));
    assertThat(test.countSentences(), is(equalTo(39L)));
    assertThat(test.countAnnotations(Annotation.Source.GOLD), is(equalTo(10L)));
  }
  
  @Test
  public void shouldReadCorrectAnnotationBoundariesFromRealWorldExample() throws Exception {
    Dataset test = new ChoiDatasetReader().read(CHOI_DATASET_EXAMPLE);
    Document document = test.getDocument(0).get();
    ArrayList<Sentence> sentences = Lists.newArrayList(document.getSentences());
    Collection<SectionAnnotation> annotationCollection = document.getAnnotations(Annotation.Source.GOLD, SectionAnnotation.class);
    ArrayList<SectionAnnotation> annotations = Lists.newArrayList(annotationCollection);

    checkSectionBoundaries(annotations.get(0), sentences.get(0), sentences.get(3));
    checkSectionBoundaries(annotations.get(1), sentences.get(4), sentences.get(8));
    checkSectionBoundaries(annotations.get(2), sentences.get(9), sentences.get(12));
    checkSectionBoundaries(annotations.get(3), sentences.get(13), sentences.get(15));
    checkSectionBoundaries(annotations.get(4), sentences.get(16), sentences.get(20));
    checkSectionBoundaries(annotations.get(5), sentences.get(21), sentences.get(24));
    checkSectionBoundaries(annotations.get(6), sentences.get(25), sentences.get(27));
    checkSectionBoundaries(annotations.get(7), sentences.get(28), sentences.get(32));
    checkSectionBoundaries(annotations.get(8), sentences.get(33), sentences.get(35));
    checkSectionBoundaries(annotations.get(9), sentences.get(36), sentences.get(38));
  }

  private void checkSectionBoundaries(SectionAnnotation section, Sentence beginSentence, Sentence endSentence) {
    assertThat(section.getBegin(), is(equalTo(beginSentence.getBegin())));
    assertThat(section.getEnd(), is(equalTo(endSentence.getEnd())));
  }

}