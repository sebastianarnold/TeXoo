package de.datexis.preprocess;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;
import static org.hamcrest.Matchers.*;

import de.datexis.common.Resource;

import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.Span;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;


public class SentenceDetectorMENLTest {

  @Test
  public void sentPosDetectShouldBeThreadSafe() throws IOException {
    List<String> lines = Files.readAllLines(Resource.fromJAR("humanismWikipediaAbstract.txt").getPath());
    SentenceModel sentenceModel = new SentenceModel(Resource.fromJAR("openNLP/en-sent.bin").getInputStream());
    SentenceDetectorMENL sentenceDetectorMENL = new SentenceDetectorMENL(sentenceModel);

    List<Span[]> sentences = lines.stream().parallel().map(s -> sentenceDetectorMENL.sentPosDetect(s))
                                .collect(Collectors.toList());
    assertThat(sentences, is(not(empty())));
  }
}