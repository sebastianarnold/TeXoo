package de.datexis.cdv.eval;

import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.index.EntityIndex;
import de.datexis.cdv.model.EntityAspectAnnotation;
import de.datexis.cdv.retrieval.EntityAspectQueryAnnotation;
import de.datexis.common.AnnotationHelpers;
import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.retrieval.model.ScoredResult;
import org.nd4j.linalg.primitives.Counter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class CDVErrorAnalysis {
  
  protected final static Logger log = LoggerFactory.getLogger(CDVErrorAnalysis.class);
  
  public static void evaluateFalsePredictions(Collection<Query> queries, Dataset corpus, EntityIndex entityIndex, AspectIndex aspectIndex, Resource outputPath) throws IOException {
  
    try(BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath.resolve("eval-" + corpus.getName() + "-errors.tsv").toFile()))) {
    
      int qid = 0;
      String heading = "qid\tquery\tquery_entity\tquery_entity_id\tquery_aspect\tquery_aspect_heading\tnum_gold\tgold_rank\t" +
        "gold_doc_id\tgold_passage_id\tgold_entity\tgold_entity_id\tgold_aspect\tgold_heading\tgold_doc_length\tgold_confidence\tgold_passage_text\t" +
        "pred_doc_id\tpred_passage_id\tpred_entity\tpred_entity_id\tpred_aspect\tpred_heading\tpred_doc_length\tpred_confidence\tpred_passage_text\n";
      writer.write(heading);
      StringBuilder out;
      for(Query q : queries) {
  
        if(q.getId() == null) q.setId(Integer.toString(qid++));
  
        // expected results (might be relevant or non-relevant) in relevance order
        Collection<Result> expected = q.getResults(Annotation.Source.GOLD, Result.class);
        // predicted results in score order
        List<ScoredResult> predicted = q.getResults(Annotation.Source.PRED, ScoredResult.class);
  
        // assign ranks to predictions and initialize them as not relevant
        AtomicInteger rank = new AtomicInteger(0);
        predicted.stream().forEach(pred -> {
          pred.setRank(rank.incrementAndGet());
          pred.setRelevance(0);
        });
  
        // match all expected results to assign relevance and scores
        for(Result exp : expected) {
          predicted.stream().forEach(pred -> {
            if(pred.matches(exp)) {
              pred.setRelevant(pred.isRelevant() || exp.isRelevant()); // update only if neccessary
              pred.setRelevance(Math.max(pred.getRelevance(), exp.getRelevance()));
              exp.setRank(pred.getRank());
              exp.setConfidence(pred.getConfidence());
            }
          });
        }
  
        List<? extends Result> relevant = predicted.stream()
          .filter(pred -> pred.isRelevant())
          .collect(Collectors.toList());
  
        List<String> relevantRanks = relevant.stream()
          .map(r -> Integer.toString(r.getRank()))
          .collect(Collectors.toList());
  
        // print query
        EntityAspectQueryAnnotation qann = q.getAnnotation(EntityAspectQueryAnnotation.class);
        for(Result exp : expected) {
          // FIXME: multiple results!
          ScoredResult pred = predicted.get(0);
          if(pred.isRelevant()) continue;
          EntityAspectAnnotation eann = AnnotationHelpers.streamAnnotationsForSpan(exp.getDocumentRef(), Annotation.Source.GOLD, EntityAspectAnnotation.class, exp).findFirst().get();
          EntityAspectAnnotation pann = AnnotationHelpers.streamAnnotationsForSpan(pred.getDocumentRef(), Annotation.Source.GOLD, EntityAspectAnnotation.class, pred).findFirst().get();
          out = new StringBuilder();
          // query
          out.append(q.getId()).append("\t"); // qid
          out.append(q.getText()).append("\t"); // query
          out.append(qann.getEntity()).append("\t"); // query_entity
          out.append(qann.getEntityId()).append(qann.getEntityId() != null && entityIndex.lookup(qann.getEntityId()) != null ? " *" : "").append("\t");// query_entity_id
          out.append(qann.getAspect()).append("\t");// query_aspect
          out.append(qann.getAspectHeading()).append("\t");// query_aspect_heading
          out.append(relevantRanks.size()).append("\t"); // num_matches
          out.append(exp.getRank()).append("\t"); // gold_rank
          out.append(exp.getDocumentRef().getId()).append("\t"); // gold_doc_id
          out.append(eann.getId()).append("\t"); // gold_passage_id
          out.append(eann.getEntity()).append("\t"); // gold_entity
          out.append(eann.getEntityId()).append("\t");// gold_entity_id
          out.append(eann.getAspect()).append("\t");// gold_aspect
          out.append(eann.getHeading()).append("\t");// gold_heading
          out.append(exp.getDocumentRef().countAnnotations(Annotation.Source.GOLD, EntityAspectAnnotation.class)).append("\t");// gold_doc_length
          out.append(fDbl(exp.getConfidence())).append("\t"); // gold_confidence
          out.append(exp.getDocumentRef().getText(eann).replaceAll("\n", " ")).append("\t");// gold_passage_text
          out.append(pred.getDocumentRef().getId()).append("\t"); // pred_doc_id
          out.append(pann.getId()).append("\t"); // pred_passage_id
          out.append(pann.getEntity()).append(pann.getEntity() != null && pann.getEntity().equalsIgnoreCase(qann.getEntity()) ? " *" : "").append("\t"); // pred_entity
          out.append(pann.getEntityId()).append(pann.getEntityId() != null && pann.getEntityId().equalsIgnoreCase(qann.getEntityId()) ? " *" : "").append("\t");// pred_entity_id
          out.append(pann.getAspect()).append(pann.getAspect() != null && pann.getAspect().equalsIgnoreCase(qann.getAspect()) ? " *" : "").append("\t");// pred_aspect
          out.append(pann.getHeading()).append("\t");// pred_heading
          out.append(pred.getDocumentRef().countAnnotations(Annotation.Source.GOLD, EntityAspectAnnotation.class)).append("\t");// pred_doc_length
          out.append(fDbl(pred.getConfidence())).append("\t"); // pred_confidence
          out.append(pred.getDocumentRef().getText(pann).replaceAll("\n", " ")).append("\t");// pred_passage_text
          writer.write(out.append("\n").toString());
          if(pred.isRelevant()) {
            if(!eann.matches(pann)) log.error("Found relevant passage that didn't match expected: {} != {}", eann.toString(), pann.toString());
          } else {
            if(eann.matches(pann)) log.error("Found irrelevant passage that matched expected: {} == {}", eann.toString(), pann.toString());
          }
        }
      }
    }
  }
  
  public static void evaluateErrorStatistics(Collection<Query> queries, Dataset corpus, EntityIndex entityIndex, AspectIndex aspectIndex, Resource outputPath) throws IOException {
 
    double all_count = 0, pos_count = 0, neg_count = 0;
    double pass_match = 0, pass_docmatch = 0, pass_mismatch = 0;
    double ea_match = 0, ea_entitymatch = 0, ea_aspectmatch = 0, ea_mismatch = 0;
    double lookup_match = 0, lookup_matchunresolved = 0, lookup_mismatchresolved = 0, lookup_mismatch = 0;
    double all_entity = 0, pos_entity = 0, neg_entity = 0,
      all_aspect = 0, pos_aspect = 0, neg_aspect = 0,
      all_docmismatch = 0, pos_docmismatch = 0, neg_docmismatch = 0,
      all_notresolved = 0, pos_notresolved = 0, neg_notresolved = 0,
      all_question = 0, pos_question = 0, neg_question = 0,
      all_wordspass = 0, pos_wordspass = 0, neg_wordspass = 0,
      all_sentspass = 0, pos_sentspass = 0, neg_sentspass = 0,
      all_passdoc = 0, pos_passdoc = 0, neg_passdoc = 0;
    
    for(Query q : queries) {
  
      // expected results (might be relevant or non-relevant) in relevance order
      Collection<Result> expected = q.getResults(Annotation.Source.GOLD, Result.class);
      // predicted results in score order
      List<ScoredResult> predicted = q.getResults(Annotation.Source.PRED, ScoredResult.class);
  
      // assign ranks to predictions and initialize them as not relevant
      AtomicInteger rank = new AtomicInteger(0);
      predicted.stream().forEach(pred -> {
        pred.setRank(rank.incrementAndGet());
        pred.setRelevance(0);
      });
  
      // match all expected results to assign relevance and scores
      for(Result exp : expected) {
        predicted.stream().forEach(pred -> {
          if(pred.matches(exp)) {
            pred.setRelevant(exp.isRelevant());
            pred.setRelevance(exp.getRelevance());
            exp.setRank(pred.getRank());
            exp.setConfidence(pred.getConfidence());
          }
        });
      }
  
      List<? extends Result> relevant = predicted.stream()
        .filter(pred -> pred.isRelevant())
        .collect(Collectors.toList());
  
      List<String> relevantRanks = relevant.stream()
        .map(r -> Integer.toString(r.getRank()))
        .collect(Collectors.toList());
  
      // top-1 result
      ScoredResult pred = predicted.get(0);
      EntityAspectQueryAnnotation qann = q.getAnnotation(EntityAspectQueryAnnotation.class);
      EntityAspectAnnotation pann = AnnotationHelpers.streamAnnotationsForSpan(pred.getDocumentRef(), Annotation.Source.GOLD, EntityAspectAnnotation.class, pred).findFirst().get();
      if(!pred.isRelevant()) neg_count++;
      else pos_count++;
      all_count++;
  
      double sentspass = 0, wordspass = 0, passdoc = 0;
      double count = 0;
      boolean docmatch = false;
      for(Result exp : expected) {
        EntityAspectAnnotation eann = AnnotationHelpers.streamAnnotationsForSpan(exp.getDocumentRef(), Annotation.Source.GOLD, EntityAspectAnnotation.class, exp).findFirst().get();
        List<Sentence> sents = exp.getDocumentRef().streamSentencesInRange(eann.getBegin(), eann.getEnd(), true).collect(Collectors.toList());
        passdoc += exp.getDocumentRef().countAnnotations(Annotation.Source.GOLD, EntityAspectAnnotation.class);
        sentspass += sents.size();
        wordspass += sents.stream().mapToInt(s -> s.countTokens()).sum();
        if(exp.getDocumentRef().equals(pred.getDocumentRef())) docmatch = true;
        count++;
      }
      
      // statistics
      if(!pred.isRelevant()) {
        neg_passdoc += passdoc / count;
        neg_sentspass += sentspass / count;
        neg_wordspass += wordspass / count;
      } else {
        pos_passdoc += passdoc / count;
        pos_sentspass += sentspass / count;
        pos_wordspass += wordspass / count;
      }
      all_passdoc += passdoc / count;
      all_sentspass += sentspass / count;
      all_wordspass += wordspass / count;

      // document mismatch
      if(pred.isRelevant()) pass_match++;
      else if(docmatch) pass_docmatch++;
      else pass_mismatch++;
      
      if(!docmatch) {
        if(!pred.isRelevant()) neg_docmismatch++;
        else pos_docmismatch++;
        all_docmismatch++;
      }
      
      // entity / aspect mismatch
      boolean entitymatch = false, aspectmatch = false;
      if(qann.getEntityId() != null && qann.getEntityId().equals(pann.getEntityId())) {
        entitymatch = true;
      } else if(qann.getEntity() != null && qann.getEntity().equalsIgnoreCase(pann.getEntity())) {
        entitymatch = true;
      }
      if(qann.getAspect().equals(pann.getAspect())) {
        aspectmatch = true;
      }
  
      if(entitymatch && aspectmatch) ea_match++;
      else if(entitymatch) ea_entitymatch++;
      else if(aspectmatch) ea_aspectmatch++;
      else ea_mismatch++;
      
      if(!entitymatch) {
        if(!pred.isRelevant()) neg_entity++;
        else pos_entity++;
        all_entity++;
      }
      if(!aspectmatch) {
        if(!pred.isRelevant()) neg_aspect++;
        else pos_aspect++;
        all_aspect++;
      }
      
      // entity lookup error
      boolean resolved = false;
      if(qann.getEntityId() != null && entityIndex.lookup(qann.getEntityId()) != null) {
        resolved = true;
      }
  
      if(pred.isRelevant() && resolved) lookup_match++;
      else if(pred.isRelevant() && !resolved) lookup_matchunresolved++;
      else if(resolved) lookup_mismatchresolved++;
      else lookup_mismatch++;
      if(!resolved) {
        if(!pred.isRelevant()) neg_notresolved++;
        else pos_notresolved++;
        all_notresolved++;
      }
  
    }
  
    StringBuilder out = new StringBuilder();
    // query
    out.append("\tALL\tPOS\tNEG\n");
    out.append("count\t")                .append(all_count).append("\t").append(pos_count).append("\t").append(neg_count).append("\n");
    out.append("entity mismatch\t")      .append(fDbl(all_entity / all_count)).append("\t").append(fDbl(pos_entity / pos_count)).append("\t").append(fDbl(neg_entity / neg_count)).append("\n");
    out.append("aspect mismatch\t")      .append(fDbl(all_aspect / all_count)).append("\t").append(fDbl(pos_aspect / pos_count)).append("\t").append(fDbl(neg_aspect / neg_count)).append("\n");
    out.append("document mismatch\t")    .append(fDbl(all_docmismatch / all_count)).append("\t").append(fDbl(pos_docmismatch / pos_count)).append("\t").append(fDbl(neg_docmismatch / neg_count)).append("\n");
    out.append("entity not resolved\t")  .append(fDbl(all_notresolved / all_count)).append("\t").append(fDbl(pos_notresolved / pos_count)).append("\t").append(fDbl(neg_notresolved / neg_count)).append("\n");
    // TODO: question
    out.append("words/passage\t")        .append(fDbl1(all_wordspass / all_count)).append("\t").append(fDbl1(pos_wordspass / pos_count)).append("\t").append(fDbl1(neg_wordspass / neg_count)).append("\n");
    out.append("sents/passage\t")        .append(fDbl1(all_sentspass / all_count)).append("\t").append(fDbl1(pos_sentspass / pos_count)).append("\t").append(fDbl1(neg_sentspass / neg_count)).append("\n");
    out.append("passages/doc\t")         .append(fDbl1(all_passdoc / all_count)).append("\t").append(fDbl1(pos_passdoc / pos_count)).append("\t").append(fDbl1(neg_passdoc / neg_count)).append("\n");
  
    out.append("\n\npassage matching:\n");
    out.append("passage matched\t").append(fDbl(pass_match / all_count)).append("\n");
    out.append("document matched\t").append(fDbl(pass_docmatch / all_count)).append("\n");
    out.append("passage mismatched\t").append(fDbl(pass_mismatch / all_count)).append("\n");
  
    out.append("\n\nentity/aspect matching:\n");
    out.append("entity/aspect matched\t").append(fDbl(ea_match / all_count)).append("\n");
    out.append("entity matched\t").append(fDbl(ea_entitymatch / all_count)).append("\n");
    out.append("aspect matched\t").append(fDbl(ea_aspectmatch / all_count)).append("\n");
    out.append("mismatch\t").append(fDbl(ea_mismatch / all_count)).append("\n");
  
    out.append("\n\nentity resolving:\n");
    out.append("positive resolved\t").append(fDbl(lookup_match / all_count)).append("\n");
    out.append("positive unseen\t").append(fDbl(lookup_matchunresolved / all_count)).append("\n");
    out.append("negative resolved\t").append(fDbl(lookup_mismatchresolved / all_count)).append("\n");
    out.append("negative unseen\t").append(fDbl(lookup_mismatch / all_count)).append("\n");
    
    System.out.println(out.toString());
  
    try(BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath.resolve("eval-" + corpus.getName() + "-errorstats.tsv").toFile()))) {
      writer.write(out.append("\n").toString());
      writer.flush();
    }
    
  }
  
  
  public static void evaluateSourcePerformance(Collection<Query> queries, Dataset corpus, EntityIndex entityIndex, AspectIndex aspectIndex, Resource outputPath) throws IOException {
  
    Counter<String> sourcesTP1 = new Counter<>();
    Counter<String> sourcesTP10 = new Counter<>();
    Counter<String> sourcesCount = new Counter<>();
  
    Counter<String> aspectsTP1 = new Counter<>();
    Counter<String> aspectsTP10 = new Counter<>();
    Counter<String> aspectsCount = new Counter<>();
  
    int K = 10;
    
    for(Query q : queries) {
    
      // expected results (might be relevant or non-relevant) in relevance order
      Collection<Result> expected = q.getResults(Annotation.Source.GOLD, Result.class);
      // predicted results in score order
      List<ScoredResult> predicted = q.getResults(Annotation.Source.PRED, ScoredResult.class);
    
      // assign ranks to predictions and initialize them as not relevant
      AtomicInteger rank = new AtomicInteger(0);
      predicted.stream().forEach(pred -> {
        pred.setRank(rank.incrementAndGet());
        pred.setRelevance(0);
      });
    
      // match all expected results to assign relevance and scores
      for(Result exp : expected) {
        predicted.stream().forEach(pred -> {
          if(pred.matches(exp)) {
            pred.setRelevant(pred.isRelevant() || exp.isRelevant());
            pred.setRelevance(exp.getRelevance());
            exp.setRank(pred.getRank());
            exp.setConfidence(pred.getConfidence());
          }
        });
      }
    
      List<? extends Result> relevant = predicted.stream()
        .filter(pred -> pred.isRelevant())
        .collect(Collectors.toList());
    
      List<String> relevantRanks = relevant.stream()
        .map(r -> Integer.toString(r.getRank()))
        .collect(Collectors.toList());
    
      // top-1 result
      ScoredResult pred = predicted.get(0);
      // top-10 result
      List<ScoredResult> top10 = predicted.stream()
        .limit(10)
        .collect(Collectors.toList());
      
      for(Result exp : expected) {
        String source = exp.getDocumentRef().getType();
        if(source == null) source = "UNK";
        String aspect = q.getAnnotation(EntityAspectQueryAnnotation.class).getAspect();
        sourcesCount.incrementCount(source, 1);
        aspectsCount.incrementCount(aspect, 1);
        if(pred.isRelevant()) {
          sourcesTP1.incrementCount(source, 1);
          aspectsTP1.incrementCount(aspect, 1);
        }
        for(ScoredResult pre : top10) {
          if(pre.isRelevant() && pre.matches(exp)) {
            sourcesTP10.incrementCount(source, 1);
            aspectsTP10.incrementCount(aspect, 1);
          }
        }
      }
    }
  
  
    try(BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath.resolve("eval-" + corpus.getName() + "-sourcestats.tsv").toFile()))) {
      System.out.println("source\tTotal\tTP@1\tTP@10");
      writer.write("source\tTotal\tTP@1\tTP@10\n");
      for(String source : sourcesCount.keySetSorted()) {
        System.out.println(source + "\t" + sourcesCount.getCount(source) + "\t" + sourcesTP1.getCount(source) + "\t" + sourcesTP10.getCount(source));
        writer.write(source + "\t" + sourcesCount.getCount(source) + "\t" + sourcesTP1.getCount(source) + "\t" + sourcesTP10.getCount(source) + "\n");
      }
      writer.flush();
    }
  
    System.out.println();
    
    try(BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath.resolve("eval-" + corpus.getName() + "-aspectstats.tsv").toFile()))) {
      System.out.println("aspect\tTotal\tTP@1\tTP@10");
      writer.write("aspect\tTotal\tTP@1\tTP@10\n");
      for(String aspect : aspectsCount.keySetSorted()) {
        System.out.println(aspect + "\t" + aspectsCount.getCount(aspect) + "\t" + aspectsTP1.getCount(aspect) + "\t" + aspectsTP10.getCount(aspect));
        writer.write(aspect + "\t" + aspectsCount.getCount(aspect) + "\t" + aspectsTP1.getCount(aspect) + "\t" + aspectsTP10.getCount(aspect) + "\n");
      }
      writer.flush();
    }
    
  }
  
  protected static String fDbl1(double d) {
    return String.format(Locale.ROOT, "%.1f", d);
  }
  
  protected static String fDbl(double d) {
    return String.format(Locale.ROOT, "%.2f", d * 100);
  }
  
}
