package de.datexis.models.index;

import de.datexis.common.WordHelpers;
import de.datexis.encoder.impl.BagOfWordsEncoder;
import de.datexis.models.index.impl.LuceneWordIndex;
import java.util.Arrays;
import java.util.List;
import static org.junit.Assert.*;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class LuceneWordIndexTest {

  protected final static Logger log = LoggerFactory.getLogger(LuceneWordIndexTest.class);

  protected static String[] diseases = new String[] {
    "Abdominal Aortic Aneurysm",
    "Acanthamoeba Infection",
    "ACE (Adverse Childhood Experiences)",
    "Acinetobacter Infection",
    "Acquired Immune Deficiency Syndrome (AIDS)",
    "Acquired Immunodeficiency Syndrome (AIDS)",
    "Adenovirus Infection",
    "Adenovirus Vaccination",
    "ADHD (Attention Deficit/Hyperactivity Disorder)",
    "Adult Vaccinations",
    "Adverse Childhood Experiences (ACE)",
    "AFib, AF (Atrial fibrillation)",
    "African Trypanosomiasis",
    "Agricultural Safety",
    "AHF (Alkhurma hemorrhagic fever)",
    "AIDS (Acquired Immune Deficiency Syndrome)",
    "AIDS (Acquired Immunodeficiency Syndrome)"
  };
      
  @Test
  public void testWordIndex() {
    
    WordIndex index = new LuceneWordIndex(Arrays.asList(diseases));
    
    List<String> result;
    
    result = index.queryText("Abdominal Aortic Aneurysm", 3);
    assertEquals(1, result.size());
    assertEquals("Abdominal Aortic Aneurysm", result.get(0));
    
    result = index.queryText("abdominal aortic aneurysm", 3);
    assertEquals(1, result.size());
    assertEquals("Abdominal Aortic Aneurysm", result.get(0));
    
    result = index.queryText("Aortic Aneurysm", 3);
    assertEquals(1, result.size());
    assertEquals("Abdominal Aortic Aneurysm", result.get(0));
    
    result = index.queryText("Abdominal Aneurysm", 3);
    assertEquals(1, result.size());
    assertEquals("Abdominal Aortic Aneurysm", result.get(0));
    
    result = index.queryText("Aneurysm Aortic", 3);
    assertEquals(1, result.size());
    assertEquals("Abdominal Aortic Aneurysm", result.get(0));
    
    result = index.queryText("ACE", 3);
    assertEquals(2, result.size());
    
    result = index.queryText("AIDS", 10);
    assertEquals(4, result.size());
    
    result = index.queryText("AIDS", 1);
    assertEquals(1, result.size());
    
    result = index.queryText("Fever", 3);
    assertEquals(1, result.size());
    assertEquals("AHF (Alkhurma hemorrhagic fever)", result.get(0));
    
    result = index.queryText("Husten", 3);
    assertEquals(0, result.size());
    
    result = index.queryText("Husten", 3);
    assertEquals(0, result.size());
    
    result = index.queryPrefixText("A", 100);
    assertEquals(diseases.length, result.size());
    
    result = index.queryPrefixText("B", 3);
    assertEquals(0, result.size());
    
    result = index.queryPrefixText("aid", 3);
    assertEquals(2, result.size()); // prefix search only on complete strings!
    
    result = index.queryPrefixText("Agricultural Sa", 3);
    assertEquals(1, result.size());
    assertEquals("Agricultural Safety", result.get(0));
    
    result = index.queryPrefixText("Vacc", 3);
    assertEquals(0, result.size()); // prefix search only on complete strings!
    
  }
  
  @Test
  public void testEncoderIndex() {
    
    BagOfWordsEncoder bow = new BagOfWordsEncoder();
    bow.trainModel(Arrays.asList(diseases), 0, WordHelpers.Language.EN);
    WordIndex index = new LuceneWordIndex(bow);
    
    List<String> result;
    
    result = index.queryText("Abdominal Aortic Aneurysm", 3);
    assertEquals(0, result.size()); // bag-of-words reduces to lowercase single words!
    
    result = index.queryText("abdominal", 3);
    assertEquals(1, result.size());
    assertEquals("abdominal", result.get(0));
    
    result = index.queryText("Aortic", 3);
    assertEquals(1, result.size());
    assertEquals("aortic", result.get(0));
    
    result = index.queryText("Aneurysm", 3);
    assertEquals(1, result.size());
    assertEquals("aneurysm", result.get(0));
    
    result = index.queryText("ACE", 3);
    assertEquals(1, result.size()); // bag-of-words removes duplicates
    assertEquals("ace", result.get(0));
    
    result = index.queryText("AIDS", 10);
    assertEquals(1, result.size()); // bag-of-words removes duplicates
    assertEquals("aids", result.get(0));
    
    result = index.queryText("Fever", 3);
    assertEquals(1, result.size());
    
    result = index.queryText("Husten", 3);
    assertEquals(0, result.size());
    
    result = index.queryPrefixText("A", 100);
    assertEquals(20, result.size());
    
    result = index.queryPrefixText("B", 3);
    assertEquals(0, result.size());
    
    result = index.queryPrefixText("aid", 3);
    assertEquals(1, result.size());
    assertEquals("aids", result.get(0));
    
    result = index.queryPrefixText("Agricultural Sa", 3);
    assertEquals(0, result.size());
    
    result = index.queryPrefixText("Vacc", 3);
    assertEquals(2, result.size());
    
  }
  
}
