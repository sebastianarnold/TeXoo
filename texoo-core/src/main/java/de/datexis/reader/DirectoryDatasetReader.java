package de.datexis.reader;

import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.preprocess.DocumentFactory;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads all files from a directory into a Dataset.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class DirectoryDatasetReader {

  protected final static Logger log = LoggerFactory.getLogger(DirectoryDatasetReader.class);
  
  public static Dataset readDatasetFromDirectory(Resource path, String pattern) throws IOException {
    log.info("Reading Documents from {}", path.toString());
    Dataset data = new Dataset(path.getPath().getFileName().toString());
    Files.walk(path.getPath())
        .filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))
        .filter(p -> p.getFileName().toString().matches(pattern))
        .sorted()
        .map(p -> readDocumentFromFile(Resource.fromFile(p.toString())))
        .forEach(d -> data.addDocument(d));
    return data;
  }
  
  public static Document readDocumentFromFile(Resource file) {
    try(InputStream in = file.getInputStream()) {
        CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
        BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
        String text = br.lines().collect(Collectors.joining("\n"));
        Document doc = DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);
        doc.setId(file.getFileName());
        doc.setSource(file.toString());
        return doc;
    } catch(IOException ex) {
      // IOException is now allowed in Stream
      log.error(ex.toString());
      throw new RuntimeException(ex.toString(), ex.getCause());
    }
  }
  

}
