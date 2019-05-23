package de.datexis.encoder;

import com.google.common.collect.Lists;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class EncodingHelpers {
  
  public static INDArray createTimeStepMatrix(long batchSize, long vectorSize, long timeSteps) {
    return Nd4j.zeros(DataType.FLOAT, batchSize, vectorSize, timeSteps);
  }
  
  /**
   * Put a single example column vector int o a time step matrix
   * @param matrix Full batch matrix [ batch size X vector size X time steps ]
   * @param batchIndex Index of the batch
   * @param t Index of the time step
   * @param value The vector to put into matrix [ vector size X 1 ]
   */
  public static void putTimeStep(INDArray matrix, long batchIndex, long t, INDArray value) {
    //matrix.getRow(batchIndex).getColumn(t).assign(vec); // invalid operation since beta4
    //matrix.put(point(batchIndex), all(), point(t), vec); // valid operation, but had errors in beta4
    //matrix.get(point(batchIndex), all(), point(t)).assign(vec); // valid operation in beta4
    matrix.slice(batchIndex, 0).slice(t, 1).assign(value); // 25% faster
  }
  
  /**
   * Get a single example column vector from a time step matrix
   * @param matrix Full batch matrix [ batch size X vector size X time steps ]
   * @param batchIndex Index of the batch
   * @param t Index of the time step
   * @return The value as column vector [ vector size X 1 ]
   */
  public static INDArray getTimeStep(INDArray matrix, long batchIndex, long t) {
    //INDArray vec = matrix.get(point(batchIndex), all(), point(t)); // valid operation in beta4
    INDArray vec = matrix.slice(batchIndex, 0).slice(t, 1); // 25% faster
    //return vec.transpose(); // invalid operation since beta4
    return vec.reshape(matrix.size(1), 1);
  }
  
  /**
   * Encodes each element in the input and returns these vectors as matrix.
   * @param input - the Documents that should be encoded
   * @param timeStepClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  public static INDArray encodeTimeStepMatrix(List<? extends Span> input, IEncoder encoder, int maxTimeSteps, Class<? extends Span> timeStepClass) {

    INDArray encoding = Nd4j.zeros(DataType.FLOAT, input.size(), encoder.getEmbeddingVectorSize(), maxTimeSteps);
    Span example;

    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {

      example = input.get(batchIndex);

      List<? extends Span> spansToEncode = Collections.EMPTY_LIST;
      if(example instanceof Document && timeStepClass == Token.class) spansToEncode = Lists.newArrayList(((Document)example).getTokens());
      else if(example instanceof Document && timeStepClass == Sentence.class) spansToEncode = Lists.newArrayList(((Document)example).getSentences());
      else if(example instanceof Sentence && timeStepClass == Token.class) spansToEncode = Lists.newArrayList(((Sentence)example).getTokens());
      else if(example instanceof Sentence && timeStepClass == Sentence.class) spansToEncode = Lists.newArrayList(((Sentence)example));

      for(int t = 0; t < spansToEncode.size() && t < maxTimeSteps; t++) {
        // TODO: Encoder should encode a batch of sentences and return matrix with batchSize columns and vectorsize rows...?
        INDArray vec = encoder.encode(spansToEncode.get(t));
        //encoding.get(point(batchIndex), all(), point(t)).assign(vec);
        encoding.slice(batchIndex, 0).slice(t, 1).assign(vec); // 25% faster
      }

    }
    return encoding;
  }
  
  public static INDArray encodeBatchMatrix(List<? extends Span> input, IEncoder encoder) {
    
    INDArray encoding = Nd4j.zeros(DataType.FLOAT, input.size(), encoder.getEmbeddingVectorSize());
    Span example;
    
    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      example = input.get(batchIndex);
      INDArray vec = encoder.encode(example);
      encoding.get(point(batchIndex), all()).assign(vec);
    }
    
    return encoding;
    
  }
  
}
