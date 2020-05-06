package de.datexis.cdv.retrieval;

import de.datexis.cdv.index.DocumentIndex;
import de.datexis.cdv.index.QueryIndex;
import de.datexis.cdv.model.EntityAspectAnnotation;
import de.datexis.common.AnnotationHelpers;
import de.datexis.common.Timer;
import de.datexis.encoder.IEncoder;
import de.datexis.model.*;
import de.datexis.model.impl.PassageAnnotation;
import de.datexis.retrieval.model.RelevanceResult;
import de.datexis.retrieval.model.ScoredResult;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class QueryRunner {
  
  protected final static Logger log = LoggerFactory.getLogger(QueryRunner.class);
  
  /** Strategy to select passages from raw text. */
  public enum Strategy {
    /** Start and end passages using per-sentence thresholds */
    SENTENCE_THRESHOLD,
    /** Use predefined passages and use averaging over the spans */
    PASSAGE_RANK
  }
  
  public enum Candidates {
    /** Use entire dataset as candidates */
    ALL,
    /** Take candidates given in query results */
    GIVEN,
    /** Use index to retrieve K candidates */
    INDEX
  }
  
  public final static int NUM_CANDIDATES = 64;
  
  Dataset corpus;
  QueryIndex entityIndex, aspectIndex;
  Strategy strategy;
  DocumentIndex index;
  protected Timer timer = new Timer();
  
  public QueryRunner(Dataset corpus, QueryIndex entityIndex, QueryIndex aspectIndex) {
    this(corpus, entityIndex, aspectIndex, Strategy.SENTENCE_THRESHOLD);
  }
  
  public QueryRunner(Dataset corpus, QueryIndex entityIndex, QueryIndex aspectIndex, Strategy strategy)  {
    this.corpus = corpus;
    this.entityIndex = entityIndex;
    this.aspectIndex = aspectIndex;
    this.strategy = strategy;
    this.index = new DocumentIndex();
    try {
      index.createInMemoryIndex(corpus);
    } catch(IOException e) {
      e.printStackTrace();
    }
  }
  
  /**
   * Retrieve all Queries on the whole corpus.
   */
  public void retrieveAllQueries() {
  
  }
  
  /**
   * Retrieve all Queries with given candidate strategy.
   */
  public void retrieveAllQueries(Candidates candidateStrategy) {
    final Timer timer = new Timer();
    timer.start();
    AtomicInteger i = new AtomicInteger();
    long count = corpus.countQueries();
    //timer.setSplit("query")
    log.info("Retrieving {} queries on {} documents...", count, corpus.countDocuments());
    corpus.getQueries().stream()
      //.parallel() // we're already parallelizing over documents, which should be enough
      .forEach(q -> {
        EntityAspectQueryAnnotation ann = q.getAnnotation(EntityAspectQueryAnnotation.class);
        if(candidateStrategy.equals(Candidates.GIVEN)) retrieveQueryFromCandidates(q);
        else if(candidateStrategy.equals(Candidates.INDEX)) retrieveQueryFromIndex(q);
        else retrieveQuery(q);
        log.info("Finished query {}/{} '{}' ({}) - '{}' [{}]", i.incrementAndGet(), count, ann.getEntity(), ann.getEntityId(), ann.getAspect(), Timer.millisToLongDHMS(timer.setSplit("query")));
      });
    long elapsed = timer.getLong();
    log.info("Finished {} queries on {} documents... [{}, {}/q]", count, corpus.countDocuments(), Timer.millisToLongDHMS(elapsed), Timer.millisToLongDHMS(elapsed / count));
  }
  
  /**
   * Retrieve all Queries on the whole corpus.
   */
  public void retrieveAllQueries(long limit) {
    for(Query query : corpus.getQueries()) {
      if(limit-- <= 0) return;
      retrieveQuery(query);
    }
  }
  
  /**
   * Retrieve all Queries, each on its own Document.
   */
  public void retrieveAllQueriesPerDocument() {
    for(Query query : corpus.getQueries()) {
      retrieveQueryFromDocs(query, Collections.singleton(query.getDocumentRef()));
    }
  }
  
  /**
   * Retrieve Query on the whole corpus.
   */
  public Query retrieveQuery(Query query) {
    return retrieveQueryFromDocs(query, corpus.getDocuments());
  }
  
  /**
   * Retrieve Query only using the given candidates.
   */
  public Query retrieveQueryFromCandidates(Query query) {
    Collection<RelevanceResult> candidates = query.getResults(Annotation.Source.GOLD, RelevanceResult.class);
    candidates.addAll(query.getResults(Annotation.Source.SILVER, RelevanceResult.class));
    Set<Document> docs = new HashSet<>();
    for(Result r : candidates) {
      docs.add(r.getDocumentRef());
    }
    query = retrieveQueryFromDocs(query, docs, candidates);
    return query;
  }
  
  /**
   * Retrieve Query only using the document index.
   */
  public Query retrieveQueryFromIndex(Query query) {
    EntityAspectQueryAnnotation ann = query.getAnnotation(EntityAspectQueryAnnotation.class);
    Collection<DocumentIndex.DocumentResult> candidates = index.search(ann.getEntity(), NUM_CANDIDATES);
    List<Document> docs = candidates.stream()
      .map(cand -> corpus.getDocument(cand.documentId).get())
      .collect(Collectors.toList());
    query = retrieveQueryFromDocs(query, docs);
    return query;
  }
  
  protected Query retrieveQueryFromDocs(Query query, Collection<Document> docs) {
    return retrieveQueryFromDocs(query, docs, null);
  }
  
  protected Query retrieveQueryFromDocs(Query query, Collection<Document> docs, Collection<? extends Annotation> candidates) {
    // encode query terms once
    EntityAspectQueryAnnotation ann = query.getAnnotation(EntityAspectQueryAnnotation.class);
    INDArray qf = null, qa = null;
    if(entityIndex != null && ann.hasEntity()) {
      qf = entityIndex.lookup(ann.getEntityId() != null ? ann.getEntityId() : ann.getEntity()); // query vector lookup
      if(qf == null) {
        log.debug("fallback encoding entity '{}'", ann.getEntity());
        qf = entityIndex.encode(ann.getEntity()); // fallback encoding
      }
    }
    if(aspectIndex != null && ann.hasAspect()) {
      qa = aspectIndex.lookup(aspectIndex.getKeyPreprocessor().preProcess(ann.getAspect())); // query vector lookup, make sure the key is not split here
      if(qa == null) {
        log.error("fallback encoding aspect '{}'", ann.getAspect());
        qa = aspectIndex.encode(ann.getAspect()); // fallback encoding
      }
    }
    final INDArray qf2 = qf;
    final INDArray qa2 = qa;
    // correlate documents with query (projection)
    docs.stream()
      .parallel()
      .filter(doc -> !doc.isEmpty())
      .forEach(doc -> {
        // encode histogram
        INDArray hist = getHistogram(doc, qf2, qa2);
        retrievePassages(doc, query, hist, candidates);
      });
    return query;
  }
  
  /**
   * Retrieve Query on a single Document.
   */
  public Query retrieveQuery(Document doc, Query query) {
    // encode histogram
    INDArray hist = getHistogram(doc, query);
    // correlate document with query (projection)
    return retrievePassages(doc, query, hist);
  }
  
  protected Query retrievePassages(Document doc, Query query, INDArray hist) {
    return retrievePassages(doc, query, hist, null);
  }
  
  protected Query retrievePassages(Document doc, Query query, INDArray hist, Collection<? extends Annotation> candidates) {
    switch(this.strategy) {
      case PASSAGE_RANK: return retrievePassagesByRanking(doc, query, hist, candidates);
      default:
      case SENTENCE_THRESHOLD: return retrievePassagesByThreshold(doc, query, hist);
    }
  }
  
  /**
   * Get the histogram for a single Document.
   */
  public INDArray getHistogram(Document doc, Query query) {
    // encode query terms
    EntityAspectQueryAnnotation ann = query.getAnnotation(EntityAspectQueryAnnotation.class);
    INDArray qf = null, qa = null;
    if(entityIndex != null && ann.hasEntity()) {
      qf = entityIndex.lookup(ann.getEntityId() != null ? ann.getEntityId() : ann.getEntity()); // query vector lookup
      if(qf == null) {
        qf = entityIndex.encode(ann.getEntity()); // fallback encoding
      }
    }
    if(aspectIndex != null && ann.hasAspect()) {
      qa = aspectIndex.lookup(ann.getAspect()); // query vector lookup
      if(qa == null) {
        qa = aspectIndex.encode(ann.getAspect()); // fallback encoding
      }
    }
    return getHistogram(doc, qf, qa);
  }
  
  protected INDArray getHistogram(Document doc, INDArray qf, INDArray qa) {
    // encode histograms for document
    if(qf != null && qa != null) {
      return projectQuery(doc, qf, qa); // concatenated query
    } else if(qf != null) {
      return projectQuery(doc, qf, entityIndex);
    } else if(qa != null) {
      return projectQuery(doc, qa, aspectIndex);
    } else return null;
  }
  
  protected INDArray projectQuery(Document doc, INDArray q, IEncoder encoder) {
    // TODO: make sure we have normalized sentence vectors from attachCDVDocumentMatrix()
    // get vector for EntityIndex or AspectIndex
    INDArray vec = doc.getVector(encoder.getClass());
    //vec.diviColumnVector(vec.norm1(1)); // normalize over full document - is already done in CDVTagger.attachCDVDocumentMatrix()
    return Transforms.unitVec(q).transpose().mmul(vec).transpose();
  }
  
  /** project entity/aspect query with concatenation */
  protected INDArray projectQuery(Document doc, INDArray qf, INDArray qa) {
    // get vector for EntityIndex or AspectIndex
    INDArray vf = doc.getVector(entityIndex.getClass());
    INDArray va = doc.getVector(aspectIndex.getClass());
    INDArray q = Nd4j.vstack(Transforms.unitVec(qf), Transforms.unitVec(qa));
    // vf and va are already normalized in CDVTagger.attachCDVDocumentMatrix()
    INDArray vec = Nd4j.vstack(vf, va);
    // normalize all sentences to Unit length after stacking
    for(int i = 0; i < vec.size(1); i++) {
      INDArray sentVec = vec.getColumn(i);
      vec.getColumn(i).assign(Transforms.unitVec(sentVec));
    }
    //if(encoder.getClass().equals(EntityIndex.class))
    //  vec.diviColumnVector(vec.norm2(1));
    return Transforms.unitVec(q).transpose().mmul(vec).transpose();
  }
  
  @Deprecated
  protected INDArray mergeHistograms(INDArray hf, INDArray ha) {
    // merge focus and aspect by averaging
    // TODO: try MAX ?
    if(hf != null && ha != null) return hf.add(ha).divi(2.);
    else if(hf != null) return hf;
    else if(ha != null) return ha;
    else throw new IllegalArgumentException("Both encodings are null");
  }
  
  protected Query retrievePassagesByRanking(Document doc, Query query, INDArray hist, Collection<? extends Annotation> candidates) {
    if(candidates == null) {
      // generate candidate passages from annotations
      candidates = doc
        .streamAnnotations(Annotation.Source.GOLD, PassageAnnotation.class, true)
        .sorted()
        .map(ann -> {
          // update documentRef if it is empty for some reason
          ann.setDocumentRef(doc);
          return ann;
        })
        .collect(Collectors.toList());
    }
    for(Annotation cand : candidates) {
      if(cand.getDocumentRef() != doc) continue;
      List<Sentence> sents = AnnotationHelpers.streamSpansInRange(doc,Sentence.class, cand.getBegin(), cand.getEnd(), true).collect(Collectors.toList());
      INDArray slice = Nd4j.zeros(sents.size());
      int length = 0;
      for(Sentence s : sents) {
        int t = doc.getSentenceIndexAtPosition(s.getBegin());
        double p = hist.getDouble(t);
        slice.putScalar(length++, p);
      }
      if(length > 0) {
        //INDArray top5 = Nd4j.sort(slice, false).get(NDArrayIndex.interval(0, Math.min(5, slice.length())));
        addResult(query, doc, cand, slice.meanNumber().doubleValue());
      }
    }
    return query;
  }
  
  /** copy from BaseNDArray, which has a bug */
  public static double percentile(double percentile, INDArray arr) {
    INDArray sorted = Nd4j.sort(arr.dup(arr.ordering()), true);
    double pos = (percentile / 100.0) * (double) (sorted.length() + 1);
  
    double fposition = FastMath.floor(pos);
    int position = (int)fposition;
  
    double diff = pos - fposition;
  
    double lower = sorted.getDouble(Math.max(0, position-1)); // fix edge case
    double upper = sorted.getDouble(Math.min(position, sorted.length() - 1)); // fixe edge case
  
    return lower + diff * (upper - lower);
  }
  
  /** root mean square / geometric mean */
  public static double rms(INDArray arr) {
    double sum = 0;
    if(arr.length() == 0) return 0;
    for(int i = 0; i < arr.length(); i++) {
      sum += Math.pow(arr.getDouble(i), 2);
    }
    if(sum == 0) return 0;
    else return Math.sqrt(sum / arr.length());
  }
  
  protected void printResults(Query q) {
    // predicted results in score order
    List<Result> predicted = q.getResults(Annotation.Source.PRED, Result.class);
    int i = 0;
    for(Result r : predicted) {
      if(++i > 10) break;
      EntityAspectAnnotation ann = (EntityAspectAnnotation) r.getAnnotationRef();
      log.info(" rank {}: {} - {} ({})", i, ann.getEntity(), ann.getAspect(), r.getConfidence());
    }
  }
  
  
  protected Query retrievePassagesByThreshold(Document doc, Query query, INDArray hist) {
    
    double docMaxRelevance = hist.maxNumber().doubleValue();
    double docSumRelevance = hist.sumNumber().doubleValue();
    double docAvgRelevance = hist.meanNumber().doubleValue();
  
    // TODO: try quadratic mean?
    int t = 0;
    double thresIn = 0.8;
    double thresOut = 0.6;
    boolean inside = false;
    int begin = 0, end = 0;
    double length = 1;
    double sum = 0.;
    // state machine
    for(Sentence s : doc.getSentences()) {
      double p = hist.getDouble(t++);
      if(!inside && p >= thresIn) {
        inside = true;
        length = 1;
        sum = p;
        begin = s.getBegin();
        end = s.getEnd();
      } else if(inside && p < thresOut) {
        inside = false;
        addResult(query, doc, begin, end, sum / length);
      } else if(inside) {
        length++;
        sum += p;
        end = s.getEnd();
      }
      // print
      //System.out.println(hist.getDouble(t++) + "\t" + s.getText().trim());
    }
    if(inside) {
      addResult(query, doc, begin, end, sum / length);
    }
    return query;
  }
  
  /** add a result with free begin / end */
  public void addResult(Query q, Document doc, int begin, int end, double score) {
    ScoredResult ann = new ScoredResult(Annotation.Source.PRED, doc, begin, end);
    ann.setConfidence(score);
    ann.setScore(score);
    q.addResult(ann);
    log.trace("adding result from document '{}' with relevance {}: '{}'", doc.getTitle(), score, doc.getText(ann));
  }
  
  /** add a result that was already defined as a passage */
  public void addResult(Query q, Document doc, Annotation passage, double score) {
    ScoredResult ann = new ScoredResult(Annotation.Source.PRED, doc, passage.getBegin(), passage.getEnd());
    ann.setConfidence(score);
    ann.setScore(score);
    ann.setAnnotationRef(passage);
    q.addResult(ann);
    log.trace("adding result from document '{}' with relevance {}: '{}'", doc.getTitle(), score, doc.getText(ann));
  }
  
}
