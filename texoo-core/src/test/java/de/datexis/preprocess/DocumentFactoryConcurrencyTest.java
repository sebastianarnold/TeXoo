package de.datexis.preprocess;

import de.datexis.common.Resource;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;
import org.apache.commons.io.IOUtils;
import org.junit.Test;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import static org.hamcrest.Matchers.*;
import static org.junit.Assert.assertThat;


public class DocumentFactoryConcurrencyTest {

  @Test
  public void tokenizerShouldBeThreadSafe() throws IOException {
    List<String> lines = IOUtils.readLines(Resource.fromJAR("datasets/humanism.txt").getInputStream(), "UTF-8");
    TokenizerModel tokenModel = new TokenizerModel(Resource.fromJAR("openNLP/en-token.bin").getInputStream());
    TokenizerMENL tokenizerMENL = new TokenizerMENL(tokenModel);
    List<Span[]> tokens = lines.stream()
      .parallel()
      .map(s -> tokenizerMENL.tokenizePos(s))
      .collect(Collectors.toList());
    assertThat(tokens, is(not(empty())));
  }
  
  @Test
  public void sentPosDetectShouldBeThreadSafe() throws IOException {
    List<String> lines = IOUtils.readLines(Resource.fromJAR("datasets/humanism.txt").getInputStream(), "UTF-8");
    SentenceModel sentenceModel = new SentenceModel(Resource.fromJAR("openNLP/en-sent.bin").getInputStream());
    SentenceDetectorMENL sentenceDetectorMENL = new SentenceDetectorMENL(sentenceModel);
    List<Span[]> sentences = lines.stream()
      .parallel()
      .map(s -> sentenceDetectorMENL.sentPosDetect(s))
      .collect(Collectors.toList());
    assertThat(sentences, is(not(empty())));
  }
  
}
