package de.datexis.encoder.impl.BERTAsAService;

import com.google.common.collect.Lists;
import com.google.gson.Gson;
import de.datexis.encoder.EncodingHelpers;
import de.datexis.encoder.impl.RESTAdapter;
import de.datexis.encoder.impl.SimpleRESTEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.datavec.api.writable.DoubleWritable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.net.URL;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class BertRESTEncoder extends SimpleRESTEncoder {

  private String vectorIdentifier;
  private BertRESTAdapter adapter;

  public BertRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super(restAdapter, vectorIdentifier, Token.class);
    this.vectorIdentifier = vectorIdentifier;
    this.adapter = (BertRESTAdapter) restAdapter;
  }

  public BertRESTEncoder(RESTAdapter restAdapter) {
    super(restAdapter, Token.class);
    this.vectorIdentifier = vectorIdentifier;
    this.adapter = (BertRESTAdapter) restAdapter;
  }

  public static BertRESTEncoder create(String domain, int port, String vectorIdentifier, int embeddingDimension, int connectionTimeout, int readTimeout) {
    return new BertRESTEncoder(new BertRESTAdapter(domain, port, embeddingDimension), vectorIdentifier);
  }


  @Override
  public INDArray encodeImpl(String word) throws IOException {
    return encode(word);
  }

  @Override
  public INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
/*
    INDArray encoding = EncodingHelpers.createTimeStepMatrix((long) input.size(), this.getEmbeddingVectorSize(), (long) maxTimeSteps);
    ArrayList<double[][][]> docEmbeddings = null;
    try {
      Instant start = Instant.now();
      docEmbeddings = this.encodeDocumentsParallel(input, maxTimeSteps, encoding);
      Instant end = Instant.now();
      System.out.println("Encoded all Documents in: " + Duration.between(start, end).toMillis() + "ms");
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    Instant start = Instant.now();
    for (int i = 0; i < docEmbeddings.size(); ++i) {
      // document
      for (int j = 0; j < docEmbeddings.get(i).length; ++j) {
        // sentence
        if (docEmbeddings.get(i)[j] == null)
          continue;
        for (int t = 0; t < docEmbeddings.get(i)[j].length; ++t) {
          // token
          INDArray vec = Nd4j.create(docEmbeddings.get(i)[j][t], new long[]{getEmbeddingVectorSize(), 1});
          EncodingHelpers.putTimeStep(encoding, (long) i, (long) t, vec);
        }
      }
    }
    Instant end = Instant.now();
    System.out.println("Put Embeddings in Encoding-Matrix in: " + Duration.between(start, end).toMillis() + "ms");
*/
/*
    for (int batchIndex = 0; batchIndex < input.size(); ++batchIndex) {
      Document example = (Document) input.get(batchIndex);
      List<? extends Span> spansToEncode = Collections.EMPTY_LIST;
      if (timeStepClass == Token.class) {
        spansToEncode = Lists.newArrayList(example.getTokens());
      } else if (timeStepClass == Sentence.class) {
        spansToEncode = Lists.newArrayList(example.getSentences());
      }c

      int offset = 0;
      for (int t = 0; t + offset < ((List) spansToEncode).size() && t < maxTimeSteps + offset; ++t) {
        Optional<Token> t_opt = example.getToken(t + offset);
        Token tok = t_opt.get();
        if (tok.isEmpty()) {
          offset++;
          continue;
        }
        try {
          INDArray vec = tok.getVector(this.vectorIdentifier);
          EncodingHelpers.putTimeStep(encoding, (long) batchIndex, (long) t, vec);
        } catch (Exception e) {
          System.out.println("Error encoding doc: " + tok.getDocumentRef().getId());
          throw e;
        }
      }
    }
*/
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

  public INDArray encodeDocumentsParallelNoTokenization(Collection<Document> documents, int maxSequenceLength) {
    INDArray encoding = EncodingHelpers.createTimeStepMatrix((long) documents.size(), this.getEmbeddingVectorSize(), (long) maxSequenceLength);
    List<BertNonTokenizedResponse> responses = documents.parallelStream().map(d -> {
      try {
        return this.adapter.simpleRequestNonTokenized(d);
      } catch (IOException e) {
        e.printStackTrace();
      }
      return null;
    }).sorted(Comparator.comparingInt(b -> b.id)).collect(Collectors.toList());

    int docIndex = 0;
    for (BertNonTokenizedResponse resp : responses) {
      for (int i = 0; i < resp.result.length && i < maxSequenceLength; ++i) {
        INDArray vec = Nd4j.create(resp.result[i], new long[]{getEmbeddingVectorSize(), 1});
        EncodingHelpers.putTimeStep(encoding, (long) docIndex, (long) i, vec);
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
        return this.adapter.simpleRequest(req);
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

  @Override
  public INDArray encodeImpl(Span span) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void encodeEachImpl(Sentence input) throws IOException {
    encodeEach1D(input.getTokens());
  }

  @Override
  public void encodeEachImpl(Document input) throws IOException {
    List<List<Token>> inputs = getTokensOfSentencesOfDocument(input);
    for (int i = 0; i < inputs.size(); ++i) {
      inputs.set(i, inputs.get(i).subList(0, Math.min(inputs.get(i).size() - 1, 509)));
    }
    inputs = inputs.stream().filter(tokens -> tokens.size() > 0).collect(Collectors.toList());

    encodeEach2D(inputs);
  }

  @Override
  public void encodeEachImpl(Collection<Document> docs) throws IOException {
    docs.forEach(d -> {
      try {
        encodeEachImpl(d);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });
    /*for (Document document : docs) {
      encodeEachImpl(document);
    }*/
  }
}
