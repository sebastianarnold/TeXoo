package de.datexis.encoder;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A Decoder decodes classes (String / int) to vectors (INDArray).
 * E.g. classifier
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public interface IDecoder {
  
  /**
	 * Get the size of the result vector
	 * @return INDArray vector length
	 */
	public long getOutputVectorSize();

  /**
   * @return {0,1}^N vector representation of the given key (NOT the dense embedding)
   */
  public INDArray decode(String key);

  /**
   * @return {0,1}^N vector representation of the given {0...1}^K embedding
   */
  public INDArray decode(INDArray vec);

  /**
   * @return {0,1}^N vector representation of the given {0...1}^N prediction (e.g. softmax)
   */
  public INDArray max(INDArray prediction);

  /**
   * @return confidence of the given {0...1}^N prediction
   */
  public double confidence(INDArray prediction);
}
