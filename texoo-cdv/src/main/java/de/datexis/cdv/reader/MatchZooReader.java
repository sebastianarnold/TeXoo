package de.datexis.cdv.reader;

import de.datexis.cdv.index.PassageIndex;
import de.datexis.common.InternalResource;
import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import de.datexis.reader.DatasetReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class MatchZooReader implements DatasetReader {
  
  protected final static Logger log = LoggerFactory.getLogger(MatchZooReader.class);
  
  @Override
  public Dataset read(Resource path) throws IOException {
    if(path instanceof InternalResource || path.isFile()) {
      Dataset data = new Dataset(path.getFileName());
      addDocumentFromFile(path, data);
      return data;
    } else if(path.isDirectory()) {
      return readDatasetFromDirectory(path, "\\.txt$");
    } else throw new FileNotFoundException("cannot open path: " + path.toString());
  }
  
  public Dataset readDatasetFromDirectory(Resource path, String pattern) throws IOException {
    log.info("Reading Documents from {}", path.toString());
    Dataset data = new Dataset(path.getPath().getFileName().toString());
    AtomicInteger progress = new AtomicInteger();
    Files.walk(path.getPath())
      .filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))
      .filter(p -> p.toString().matches(pattern))
      .forEach(p -> {
        try {
          addDocumentFromFile(Resource.fromFile(p.toString()), data);
        } catch(IOException e) {
          e.printStackTrace();
        }
        int n = progress.incrementAndGet();
        if(n % 1000 == 0) {
          double free = Runtime.getRuntime().freeMemory() / (1024. * 1024. * 1024.);
          double total = Runtime.getRuntime().totalMemory() / (1024. * 1024. * 1024.);
          log.debug("read {}k documents, memory usage {} GB", n / 1000, (int)((total-free) * 10) / 10.);
        }
      });
    return data;
  }
  
  protected abstract void addDocumentFromFile(Resource path, Dataset data) throws IOException;
  
  public static void addCandidateSamples(Dataset corpus, int numCandidates) throws IOException {
    log.info("Adding {} passage candidates to queries...", numCandidates);
    PassageIndex index = new PassageIndex();
    index.createInMemoryIndex(corpus);
    index.retrievePassageCandidates(corpus, numCandidates);
  }
  
}
