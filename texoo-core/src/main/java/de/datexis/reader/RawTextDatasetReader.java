package de.datexis.reader;

import de.datexis.common.Resource;
import de.datexis.model.Document;
import de.datexis.preprocess.DocumentFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

/**
 * Reads and parses raw text files into a Dataset.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class RawTextDatasetReader extends DirectoryDatasetReader<RawTextDatasetReader> {

  protected final static Logger log = LoggerFactory.getLogger(RawTextDatasetReader.class);
  
  protected boolean useFirstSentenceAsTitle = false;
  protected boolean isTokenized = false;
  protected boolean generateUIDs = false;
  
  /**
   * Use a copy of every first sentence as Document title.
   */
  public RawTextDatasetReader withFirstSentenceAsTitle(boolean useFirstSentence) {
    this.useFirstSentenceAsTitle = useFirstSentence;
    return this;
  }
  
  /**
   * Set to TRUE if the input files are already tokenized and space-separated.
   */
  public RawTextDatasetReader withTokenizedInput(boolean isTokenized) {
    this.isTokenized = isTokenized;
    return this;
  }
  
  /**
   * Set to TRUE if the documents should be assigned incremental UIDs.
   */
  public RawTextDatasetReader withGeneratedUIDs(boolean generate) {
    this.generateUIDs = generate;
    return this;
  }
  
  /**
   * Read a single Document from file.
   */
  @Override
  public Document readDocumentFromFile(Resource file) throws IOException {
    try(InputStream in = file.getInputStream()) {
        CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
        BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
        String text = br.lines().collect(Collectors.joining("\n"));
        Document doc = isTokenized ? 
            DocumentFactory.fromTokenizedText(text) :
            DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);
        doc.setId(file.getFileName());
        doc.setSource(file.toString());
        if(useFirstSentenceAsTitle) {
          if(doc.countSentences() > 0) {
            doc.setTitle(doc.getSentence(0).getText().trim());
          } else {
            doc.setTitle("");
          }
        }
        return doc;
    }
  }

}
