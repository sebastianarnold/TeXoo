package de.datexis.encoder;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.List;

public interface Encoder {
  /**
   * Get the size of the result vector
   * @return INDArray scalar length
   */
  long getVectorSize();

  /**
   * Generate a fixed-size vector of a String
   * @param word
   * @return
   */
  INDArray encode(String word);

  /**
   * Generate a fixed-size vector of a Token
   * @param span the Span to encode
   * @return INDArray containing the encoded Token
   */
  INDArray encode(Span span);

  /**
   * Encode a fixed-size vector from multiple Tokens
   * @param spans the Spans to encode
   * @return INDArray containing all Tokens combined
   */
  INDArray encode(Iterable<? extends Span> spans);

  /**
   * Encodes each element in the input and attaches the vectors to the element.
   * Please override this if the elements of your encoders are not independent or stateful.
   * @param input - the Sentence that should be encoded
   * @param elementClass - the class of sub elements in the Sentence, e.g. Token.class
   */
  void encodeEach(Sentence input, Class<? extends Span> elementClass);

  /**
   * Encodes each element in the input and attaches the vectors to the element.
   * Please override this if the elements of your encoders are not independent or stateful.
   * @param input - the Document that should be encoded
   * @param elementClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  void encodeEach(Document input, Class<? extends Span> elementClass);

  /**
   * Encodes each element in the input and attaches the vectors to the element.
   * Please override this if the elements of your encoders are not independent or stateful.
   * Please override this if your encoder allows batches.
   * @param docs - the Documents that should be encoded
   * @param elementClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  void encodeEach(Collection<Document> docs, Class<? extends Span> elementClass);

  /**
   * Encodes each element in the input and returns these vectors as matrix.
   * Please override this if the elements of your encoders are not independent or stateful.
   *  @param input - the Document that should be encoded
   * @param timeStepClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass);
}
