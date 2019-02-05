package de.datexis.encoder;

import com.google.common.collect.Lists;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

public class EncodingHelpers {

  /**
   * Encodes each element in the input and returns these vectors as matrix.
   * @param input - the Documents that should be encoded
   * @param timeStepClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  public static INDArray encodeTimeStepMatrix(List<Document> input, IEncoder encoder, int maxTimeSteps, Class<? extends Span> timeStepClass) {

    INDArray encoding = Nd4j.zeros(input.size(), encoder.getEmbeddingVectorSize(), maxTimeSteps);
    Document example;

    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {

      example = input.get(batchIndex);

      List<? extends Span> spansToEncode = Collections.EMPTY_LIST;
      if(timeStepClass == Token.class) spansToEncode = Lists.newArrayList(example.getTokens());
      else if(timeStepClass == Sentence.class) spansToEncode = Lists.newArrayList(example.getSentences());

      for(int t = 0; t < spansToEncode.size() && t < maxTimeSteps; t++) {
        INDArray vec = encoder.encode(spansToEncode.get(t));
        encoding.getRow(batchIndex).getColumn(t).assign(vec);
      }

    }
    return encoding;
  }

}
