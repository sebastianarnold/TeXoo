package de.datexis.encoder;

import de.datexis.encoder.impl.ExampleEncoder;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

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
  public void testTimeSeriesMatrix() {
    
    long batchSize = 3;
    long vectorSize = 17;
    long timeSteps = 5;
    
    INDArray indexEncoding = Nd4j.zeros(batchSize, vectorSize, timeSteps);
    INDArray rowcolEncoding = Nd4j.zeros(batchSize, vectorSize, timeSteps);

    for(int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
      for(int t = 0; t < timeSteps; t++) {
        INDArray vec = Nd4j.rand(new long[] {vectorSize, 1});
        // we want to make sure that both these access methods provide the same results
        indexEncoding.put(new INDArrayIndex[] {point(batchIndex), all(), point(t)}, vec);
        // this one is faster, but produced an IllegalArumentException in in Dl4j-beta4-SNAPSHOT
        //rowcolEncoding.getRow(batchIndex).getColumn(t).assign(vec);
        rowcolEncoding.get(new INDArrayIndex[] {point(batchIndex), all(), point(t)}).assign(vec);
      }
    }
    
    assertEquals(indexEncoding, rowcolEncoding);
    
  }
  
}
