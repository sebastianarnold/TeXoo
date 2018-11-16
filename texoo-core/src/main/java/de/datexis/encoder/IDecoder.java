package de.datexis.encoder;

import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A Decoder converts hidden layer vectors (INDArray) to output layer vectors (INDArray).
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
	 * Generate a fixed-size vector of a String
	 * @param word
	 * @return Mx1 column vector (INDArray) containing the decoded String
	 */
	public abstract INDArray decode(String word);
  
  /**
	 * Generate a fixed-size vector of a single Span
   * @param span the Span to encode
	 * @return Mx1 column vector (INDArray) containing the decoded Span
	 */
	public abstract INDArray decode(Span span);
  
  /**
   * Encode a fixed-size vector from multiple Spans
   * @param spans the Spans to encode
   * @return Mx1 column vector (INDArray) containing all Spans combined (e.g. average)
   */
  public INDArray decode(Iterable<? extends Span> spans);
  
}
