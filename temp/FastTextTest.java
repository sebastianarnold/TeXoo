package de.datexis.encoder.impl;

import de.datexis.common.Resource;
import java.io.IOException;
import org.junit.Test;
import static org.junit.Assert.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class FastTextTest {
  
  public FastTextTest() {
  }
  
  //@Test
  public void testFastTextBinary() throws IOException {
    FastTextEncoder ft = new FastTextEncoder();
    ft.loadModel(Resource.fromFile("/home/sarnold/Library/Models/TeXoo/moblie_1.0/fasttext/", "mt-word-reps.bin"));
    // nlabels=0, ntokens=306515321, nwords=900649, size=900649,
    System.out.println(ft.getNearestNeighbours("Caddy", 20));
    System.out.println(ft.getNearestNeighbours("Touran", 20));
    System.out.println(ft.getNearestNeighbours("Truthahn", 20));
    System.out.println(ft.getNearestNeighbours("Cady", 20));
    System.out.println(ft.getNearestNeighbours("VW", 20));
    INDArray vec = ft.encode("Touran");
    System.out.println(vec.shapeInfoToString());
    System.out.println(vec.toString());
    // size = 100
  }
}
