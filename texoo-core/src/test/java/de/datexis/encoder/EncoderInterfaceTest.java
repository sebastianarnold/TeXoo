package de.datexis.encoder;

import org.junit.Test;
import static org.junit.Assert.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class EncoderInterfaceTest {
  
  public EncoderInterfaceTest() {
  }
  
  @Test
  public void testEncoders() {
    
    // TODO: find all IEncoders via reflection
    // TODO: run the test on each encoder that is loadable without model
    
    IEncoder enc = new ExampleEncoder();
    long m = enc.getEmbeddingVectorSize();
    assertTrue(m > 0);
    assertEquals(m, enc.getEmbeddingVectorSize());
    INDArray vec = enc.encode("Test");
    assertNotNull(vec);
    assertTrue(vec.isColumnVector());
    assertEquals(m, vec.rows());
    assertEquals(1, vec.columns());
    assertArrayEquals(new long[]{m, 1}, vec.shape());
    
  }
  
  @Test
  public void testDecoders() {
    
    // TODO: find all IEncoders via reflection
    // TODO: run the test on each encoder that is loadable without model
    
    IDecoder enc = new ExampleEncoder();
    long m = enc.getOutputVectorSize();
    assertTrue(m > 0);
    assertEquals(m, enc.getOutputVectorSize());
    INDArray vec = enc.decode("Test");
    System.out.println(vec.toString());
    assertNotNull(vec);
    assertTrue(vec.isColumnVector());
    assertEquals(m, vec.rows());
    assertEquals(1, vec.columns());
    assertArrayEquals(new long[]{m, 1}, vec.shape());
    
  }
  
}
