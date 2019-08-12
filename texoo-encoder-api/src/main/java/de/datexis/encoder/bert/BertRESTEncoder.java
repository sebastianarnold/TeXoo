package de.datexis.encoder.bert;

import com.google.gson.Gson;
import de.datexis.encoder.AbstractRESTEncoder;
import de.datexis.encoder.EncodingHelpers;
import de.datexis.encoder.RESTAdapter;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class BertRESTEncoder extends AbstractRESTEncoder {

  protected BertRESTEncoder() {
    super("BERT");
  }

  ;
  private int currentRequestID = 0;

  public BertRESTEncoder(RESTAdapter restAdapter) {
    super("BERT", restAdapter);
  }

  public static BertRESTEncoder create(String domain, int port, int embeddingDimension) {
    return new BertRESTEncoder(new BertRESTAdapter(domain, port, embeddingDimension));
  }

  @Override
  public INDArray encode(String word) {
    throw new UnsupportedOperationException("BERT cannotbe used to encode single words");
  }

  @Override
  public INDArray encode(Span span) {
    throw new UnsupportedOperationException("please use encodeMatrix()");
  }

  @Override
  public INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    if (timeStepClass == Sentence.class) {
      return encodeSentences(input, maxTimeSteps);
    } else {
      return encodeTokens(input, maxTimeSteps);
    }
  }

  // Token encoding
  private INDArray encodeTokens(List<Document> documents, int maxTimeSteps) {
    INDArray encoding = EncodingHelpers.createTimeStepMatrix((long) documents.size(), this.getEmbeddingVectorSize(), (long) maxTimeSteps);
    List<String> requests = new ArrayList<>();
    for (Document d : documents) {
      if (d.getSentences().size() == 0)
        continue;
      Gson gson = new Gson();
      String[] sentences = new String[Math.min(d.getSentences().size(), maxTimeSteps)];
      for (int i = 0; i < Math.min(d.getSentences().size(), maxTimeSteps); ++i) {
        sentences[i] = d.getSentence(i).getText();
      }
      BaaSRequestNonTokenized request = new BaaSRequestNonTokenized();
      request.texts = sentences;
      request.id = currentRequestID++;
      String req = gson.toJson(request);
      requests.add(req);
    }


    List<BertResponse> responses = requests.parallelStream()
      .map(req -> {
        try {
          return ((BertRESTAdapter) this.restAdapter).encodeTokens(req, maxTimeSteps);
        } catch (IOException e) {
          e.printStackTrace();
        }
        return null;
      }).filter(Objects::nonNull)
      .sorted(Comparator.comparingInt(b -> b.id))
      .collect(Collectors.toList());

    int docIndex = 0;
    // response consists of [Sentence[Token[EmbeddingId]]]
    for (BertResponse resp : responses) {
      int t = 0;
      fillMatrix:
      for (int sentenceId = 0; sentenceId < resp.result.length; ++sentenceId) {
        for (int tokenId = 0; tokenId < resp.result[sentenceId].length; ++tokenId) {
          INDArray vec = Nd4j.create(resp.result[sentenceId][tokenId], new long[]{getEmbeddingVectorSize(), 1});
          EncodingHelpers.putTimeStep(encoding, (long) docIndex, (long) t, Transforms.unitVec(vec));
          t++;
          if (t >= maxTimeSteps)
            break fillMatrix;
        }
      }
      docIndex++;
    }
    return encoding;
  }

  // Sentence encoding
  public INDArray encodeSentences(Collection<Document> documents, int maxSequenceLength) {
    INDArray encoding = EncodingHelpers.createTimeStepMatrix((long) documents.size(), this.getEmbeddingVectorSize(), (long) maxSequenceLength);

    // create requests
    List<String> requests = new ArrayList<>();
    for (Document d : documents) {
      if (d.getSentences().size() == 0)
        continue;
      Gson gson = new Gson();
      String[] sentences = new String[Math.min(d.getSentences().size(), maxSequenceLength)];
      for (int i = 0; i < Math.min(d.getSentences().size(), maxSequenceLength); ++i) {
        sentences[i] = d.getSentence(i).getText();
      }
      BaaSRequestNonTokenized request = new BaaSRequestNonTokenized();
      request.texts = sentences;
      request.id = currentRequestID++;
      String req = gson.toJson(request);
      requests.add(req);
    }

    List<BertNonTokenizedResponse> responses = requests.parallelStream()
      .map(req -> {
        try {
          return ((BertRESTAdapter) this.restAdapter).encodeSentences(req, maxSequenceLength);
        } catch (IOException e) {
          e.printStackTrace();
        }
        return null;
      }).filter(Objects::nonNull)
      .sorted(Comparator.comparingInt(b -> b.id))
      .collect(Collectors.toList());

    int docIndex = 0;
    for (BertNonTokenizedResponse resp : responses) {
      for (int i = 0; i < resp.result.length && i < maxSequenceLength; ++i) {
        INDArray vec = Nd4j.create(resp.result[i], new long[]{getEmbeddingVectorSize(), 1});
        EncodingHelpers.putTimeStep(encoding, (long) docIndex, (long) i, Transforms.unitVec(vec));
      }
      docIndex++;
    }
    return encoding;

  }
}
