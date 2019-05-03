package de.datexis.encoder;

import com.google.common.collect.Lists;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.util.Collections;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class EncodingHelpers {

  /**
   * Encodes each element in the input and returns these vectors as matrix.
   * @param input - the Documents that should be encoded
   * @param timeStepClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  public static INDArray encodeTimeStepMatrix(List<? extends Span> input, IEncoder encoder, int maxTimeSteps, Class<? extends Span> timeStepClass) {

    INDArray encoding = Nd4j.zeros(input.size(), encoder.getEmbeddingVectorSize(), maxTimeSteps);
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
        encoding.get(point(batchIndex), all(), point(t)).assign(vec);
      }

    }
    return encoding;
  }
  

}
