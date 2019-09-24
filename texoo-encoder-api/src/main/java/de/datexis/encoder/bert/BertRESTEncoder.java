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
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.Collectors;

public class BertRESTEncoder extends AbstractRESTEncoder {

  protected BertRESTEncoder() {
    super("BERT");
  };
  
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
    // check if we have all the sentence vectors already
    if(isCachingEnabled() && timeStepClass.equals(Sentence.class) && input.stream()
      .flatMap(doc -> doc.streamSentences()
        .limit(maxTimeSteps)
        .map(sent -> sent.hasVector(this.getClass())))
      .allMatch(b -> b == true)) {
      // use the implementation that returns cached vectors
      return super.encodeMatrix(input, maxTimeSteps, timeStepClass);
    }
    return encodeDocumentsParallelNoTokenization(input, maxTimeSteps);
  }

  private String documentToRequest(Document d, int id, int maxSequenceLength) {
    List<Sentence> sentences = d.getSentences();
    sentences = sentences.stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    String[][] sequences = new String[d.getSentences().size()][];
    for (int i = 0; i < sentences.size(); ++i) {
      String[] sequence = new String[Math.min(sentences.get(i).getTokens().size(), maxSequenceLength)];
      for (int j = 0; j < sentences.get(i).getTokens().size() && j < maxSequenceLength; ++j) {
        sequence[j] = sentences.get(i).getToken(j).getText();
      }
      sequences[i] = sequence;
    }
    Gson gson = new Gson();
    BaaSRequest req = new BaaSRequest();
    req.id = id;
    req.texts = sequences;
    req.is_tokenized = true;
    return gson.toJson(req);
  }

  public INDArray encodeDocumentsParallelNoTokenization(List<Document> documents, int maxSequenceLength) {
    INDArray encoding = EncodingHelpers.createTimeStepMatrix((long) documents.size(), this.getEmbeddingVectorSize(), (long) maxSequenceLength);
    List<BertNonTokenizedResponse> responses = documents.parallelStream()
      .map(d -> {
        try {
          if(d != null && d.getSentences().size() > 0)
            return ((BertRESTAdapter)this.restAdapter).simpleRequestNonTokenized(d, maxSequenceLength);
        } catch (IOException e) {
          e.printStackTrace();
          System.out.println("Error at document: " + d.getId());
        }
        return null;
      }).collect(Collectors.toList());

    int docIndex = 0;
    for (BertNonTokenizedResponse resp : responses) {
      if(resp != null) {
        for(int t = 0; t < resp.result.length && t < maxSequenceLength; ++t) {
          INDArray vec = Transforms.unitVec(Nd4j.create(resp.result[t], new long[]{getEmbeddingVectorSize(), 1}));
          EncodingHelpers.putTimeStep(encoding, (long) docIndex, (long) t, vec);
          if(isCachingEnabled()) documents.get(docIndex).getSentence(t).putVector(this.getClass(), vec);
        }
      }
      docIndex++;
    }
    return encoding;

  }


  public ArrayList<double[][][]> encodeDocumentsParallel(Collection<Document> documents, int maxSequenceLength, INDArray toFill) throws InterruptedException {
    ArrayList<double[][][]> results = new ArrayList<>();
    LinkedBlockingQueue<BertResponse> resps = new LinkedBlockingQueue<>();
    ArrayList<String> requests = new ArrayList<>();

    Instant beforeRequestGen = Instant.now();
    int id = 0;
    for (Document doc : documents) {
      requests.add(documentToRequest(doc, id, maxSequenceLength));
      id++;
    }
    Instant afterRequestGen = Instant.now();
    long requestGenDur = Duration.between(beforeRequestGen, afterRequestGen).toMillis();

    Instant beforeRequest = Instant.now();
    List<BertResponse> responses = requests.parallelStream().map(req -> {
      try {
        return ((BertRESTAdapter)this.restAdapter).simpleRequest(req);
      } catch (IOException e) {
        e.printStackTrace();
      }
      return null;
    }).sorted((Comparator.comparingInt(resp -> resp.id))).collect(Collectors.toList());

    Instant afterRequest = Instant.now();
    long requestDur = Duration.between(beforeRequest, afterRequest).toMillis();

    // remove first and last element from sequence arrays
    Instant beforeArrayGen = Instant.now();
    int docId = 0;
    for (BertResponse respons : responses) {
      double[][][] result = new double[respons.result.length][][];
      // [sentence][token][certain EmbeddingValue]
      for (int i = 0; i < respons.result.length && i < maxSequenceLength; ++i) {
        result[i] = Arrays.copyOfRange(respons.result[i], 1, respons.result[i].length - 1);
      }
      results.add(result);
      docId++;
    }
    Instant afterArrayGen = Instant.now();
    long arrayGenDur = Duration.between(beforeArrayGen, afterArrayGen).toMillis();
    System.out.println("Request generation: " + requestGenDur + "\n" + "Requests: " + requestDur + "\n" + "Array generation: " + arrayGenDur);
    return results;

  }
  
}
