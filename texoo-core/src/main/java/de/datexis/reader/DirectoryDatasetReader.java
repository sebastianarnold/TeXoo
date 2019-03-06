package de.datexis.reader;

import de.datexis.common.InternalResource;
import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Abstract implementation for a DatasetReader that reads multiple files from a directory.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class DirectoryDatasetReader<A extends DatasetReader> implements DatasetReader {
  
  protected final Logger log = LoggerFactory.getLogger(DirectoryDatasetReader.class);
  
  protected boolean randomizeDocuments = false;
  protected long limit = -1;
  
  /**
   * Use a copy of every first sentence as Document title.
   */
  public A withRandomizedDocuments(boolean randomize) {
    this.randomizeDocuments = randomize;
    return (A) this;
  }
  
  /**
   * Stop after reading a given number of documents.
   */
  public A withLimitNumberOfDocuments(long limit) {
    this.limit = limit;
    return (A) this;
  }
  
  /**
   * Read Dataset from a given directory or file.
   */
  @Override
  public Dataset read(Resource path) throws IOException {
    if(path instanceof InternalResource || path.isFile()) {
      Document doc = readDocumentFromFile(path);
      Dataset data = new Dataset(path.getFileName());
      data.addDocument(doc);
      return data;
    } else if(path.isDirectory()) {
      return readDatasetFromDirectory(path, ".+");
    } else throw new FileNotFoundException("cannot open path: " + path.toString());
  }
  
  /**
   * Read Dataset from a given directory or file.
   */
  public abstract Document readDocumentFromFile(Resource file) throws IOException;
  
  /**
   * Read Dataset from a given directory of files.
   */
  public Dataset readDatasetFromDirectory(Resource path) throws IOException {
    return readDatasetFromDirectory(path, ".+");
  }
  
  /**
   * Read Dataset from a given directory of files.
   * @param pattern REGEX pattern to match only selected file names
   */
  public Dataset readDatasetFromDirectory(Resource path, String pattern) throws IOException {
    log.info("Reading Documents from {}", path.toString());
    Dataset data = new Dataset(path.getPath().getFileName().toString());
    AtomicInteger progress = new AtomicInteger();
    Stream<Path> paths = Files.walk(path.getPath())
      .filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))
      .filter(p -> p.getFileName().toString().matches(pattern))
      .sorted();
    if(randomizeDocuments) {
      List<Path> list = paths.collect(Collectors.toList());
      Collections.shuffle(list);
      paths = list.stream();
    }
    Stream<Document> docs = paths
      .flatMap(p -> tryReadDocumentsFromFile(Resource.fromFile(p.toString())))
      .filter(d -> !d.isEmpty());
    if(limit >= 0) {
      docs = docs.limit(limit);
    }
    docs.forEach(d -> {
      long n = progress.incrementAndGet();
      data.addDocument(d);
      if(n % 1000 == 0) {
        double free = Runtime.getRuntime().freeMemory() / (1024. * 1024. * 1024.);
        double total = Runtime.getRuntime().totalMemory() / (1024. * 1024. * 1024.);
        log.debug("read {}k documents, memory usage {} GB", n / 1000, (int)((total-free) * 10) / 10.);
      }
    });
    return data;
  }
  
  /**
   * Read a Document(s) from file without IOException. Default implementation for a single Document per file.
   */
  protected Stream<Document> tryReadDocumentsFromFile(Resource path) {
    try {
      return Stream.of(readDocumentFromFile(path));
    } catch(IOException ex) {
      // IOException is now allowed in Stream
      log.error(ex.toString());
      throw new RuntimeException(ex.toString(), ex.getCause());
    }
  }
  
}
