package de.datexis.retrieval.index;


import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.impl.TrigramEncoder;
import de.datexis.model.Document;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import de.datexis.reader.RawTextDatasetReader;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

public class InMemoryIndexTest {
  
  Encoder trigram;
  List<String> headings;
  Map<String, String> entries;
  
  public InMemoryIndexTest() throws IOException {
    Document wiki = new RawTextDatasetReader().readDocumentFromFile(Resource.fromJAR("testdata/en_disease_dermatitis.txt"));
    trigram = new TrigramEncoder();
    trigram.trainModel(Collections.singleton(wiki));
    headings = new LinkedList<>();
    entries = new TreeMap<>();
    String[] splits = new String[] {"Signs and symptoms", "Causes", "Causes", "Genetics", "Causes", "Hygiene hypothesis",
      "Causes", "Allergens", "Causes", "Hard water", "Pathophysiology", "Diagnosis", "Treatments", "Treatments", "Lifestyle",
      "Treatments", "Diet", "Treatments", "Medication", "Treatments", "Light", "Epidemiology", "Research" }; // taken from en_disease_dermatitis.json
    for(String s : splits) {
      headings.add(s);
      entries.putIfAbsent(s, s);
    }
  }
  
  @Test
  public void testKeyIndex() {
    assertEquals(23, headings.size());
  
    InMemoryIndex index = new InMemoryIndex(new MinimalLowercasePreprocessor(), trigram);
    index.buildKeyIndex(headings);
  
    assertEquals(15, index.size());
    assertEquals(23, index.totalInstances());
  
    assertEquals(1, (int) index.frequency("Signs and symptoms"));
    assertEquals(0, (int) index.frequency("Signs"));
    assertEquals(5, (int) index.frequency("Causes"));
    assertEquals(5, (int) index.frequency("causes"));
    assertEquals(1, (int) index.frequency("Genetics"));
    assertEquals(5, (int) index.frequency("Treatments"));
    assertEquals(1, (int) index.frequency("Research"));
    assertEquals(0, (int) index.frequency("Distribution"));
    assertEquals(0, (int) index.frequency(""));
  
    int idx;
  
    idx = index.index("Causes");
    assertTrue(idx >= 0);
    assertEquals(idx, index.index("causes"));
    assertEquals("causes", index.key(idx));
    assertEquals(5, (int) index.frequency(idx));
    assertEquals(5. / 23., index.probability(idx), 0.00001);
    assertEquals(5. / 23., index.probability("Causes"), 0.00001);
  
    idx = index.index("Distribution");
    assertEquals(-1, idx);
    assertNull(index.key(idx)); // TODO: is NULL intended?
    assertEquals(0, (int) index.frequency(idx));
    assertEquals(0, index.probability(idx), 0.00001);
    
    assertEquals(trigram.getEmbeddingVectorSize(), index.getEmbeddingVectorSize());
  
  }
  
  @Test
  public void testVectorIndex() {
  
    TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
    InMemoryIndex index = new InMemoryIndex(preprocessor, trigram);
    index.buildKeyIndex(headings);
    index.encodeAndBuildVectorIndex(entries, true);
  
    INDArray vec, enc;
    
    vec = index.lookup("Signs and symptoms");
    enc = trigram.encode("Signs and symptoms");
    assertNotNull(vec);
    assertNotNull(enc);
    // IEncoder requires a Mx1 column vector (INDArray)
    assertFalse(enc.isRowVector());
    assertTrue(enc.isColumnVector());
    assertEquals(trigram.getEmbeddingVectorSize(), enc.rows());
    assertEquals(1, enc.columns());
    assertFalse(vec.isRowVector());
    assertTrue(vec.isColumnVector());
    assertEquals(index.getEmbeddingVectorSize(), vec.rows());
    assertEquals(1, vec.columns());
    assertEquals(Transforms.unitVec(enc), vec); // vec is normalized
    
    // test lookup
    assertEquals(trigram.getEmbeddingVectorSize(), index.getEmbeddingVectorSize());
    assertEquals(trigram.getEmbeddingVectorSize(), vec.length());
    assertTrue(vec.sumNumber().intValue() > 0);
    assertNotEquals(index.encode("signs or symptoms"), vec);
    assertEquals(vec, index.lookup("signs and symptoms ")); // key is normalized
    
    // test all keys
    for(String h : headings) {
      vec = index.lookup(h);
      assertNotNull(vec);
      assertTrue(vec.sumNumber().intValue() > 0);
      assertEquals(index.encode(h), vec);
      assertEquals(vec, index.encode(h));
    }
  
    // test null keys
    vec = index.lookup("Distribution");
    assertNull(vec);
    vec = index.lookup("");
    assertNull(vec);
    
    // test nearest neighbours
    assertEquals("signs_and_symptoms", index.find(index.lookup("Signs and symptoms")).key);
    assertEquals("signs_and_symptoms", index.find(index.encode("Signs and symptom")).key);
    assertEquals("signs_and_symptoms", index.find(index.encode("Signs or symptoms")).key);
    assertEquals("signs_and_symptoms", index.find(index.encode("symptoms and signs")).key);
    assertEquals("signs_and_symptoms", index.find(index.encode("Signs")).key);
    assertEquals("signs_and_symptoms", index.find(index.encode("symptom")).key);
    assertEquals("research", index.find(index.encode("search")).key);
    assertTrue(index.find(index.encode("Distribution"), 10).size() > 1);
    assertTrue(index.find(index.encode(""), 10).size() == 0); // index should not return 0 similarities
    //System.out.println(index.find(index.encode("")).toString());
  
  }
  
  @Test(expected=IllegalArgumentException.class)
  public void testInvalidInput() {
    TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
    InMemoryIndex index = new InMemoryIndex(preprocessor, trigram);
    index.buildKeyIndex(headings);
    index.encodeAndBuildVectorIndex(entries, true);
    index.find(Nd4j.ones(10, 1)).toString();
  }
  
  @Test
  public void testIndexSerialization() throws IOException {
  
    Resource path = Resource.createTempDirectory();
  
    TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
    InMemoryIndex index = new InMemoryIndex(preprocessor, trigram);
    index.buildKeyIndex(headings);
    index.encodeAndBuildVectorIndex(entries, true);
    index.saveModel(path, "index");
  
    InMemoryIndex index2 = new InMemoryIndex(preprocessor, trigram);
    index2.loadModel(path.resolve("index.bin"));
    
    assertEquals(index.getEmbeddingVectorSize(), index2.getEmbeddingVectorSize());
    assertEquals(index.size(), index2.size());
    assertEquals(index.totalInstances(), index2.totalInstances());
    assertEquals(index.getKeyPreprocessor().getClass(), index2.getKeyPreprocessor().getClass());
    assertEquals(index.getEncoder().getClass(), index2.getEncoder().getClass());
  
    assertEquals(index.frequency("Causes"), index2.frequency("Causes"), 0.00001);
    assertEquals(index.index("causes"), index2.index("causes"));
    assertEquals(index.probability("Causes"), index2.probability("Causes"), 0.00001);
    assertEquals(index.encode("Causes"), index2.encode("Causes"));
    assertEquals(index.lookup("Causes"), index2.lookup("Causes"));
    INDArray vec = index.lookup("Causes");
    assertEquals(index.findKey(vec), index2.findKey(vec));
    
  }
  
}
