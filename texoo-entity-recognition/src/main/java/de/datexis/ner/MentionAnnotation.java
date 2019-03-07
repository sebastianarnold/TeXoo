package de.datexis.ner;

import de.datexis.common.WordHelpers;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.model.tag.BIOESTag;
import de.datexis.model.tag.Tag;
import de.datexis.tagger.AbstractIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Annotation of a named entity mention.
 * @author sarnold, fgrimme
 */
public class MentionAnnotation extends Annotation {

  private static final Logger log = LoggerFactory.getLogger(MentionAnnotation.class);

	public static class Type {
		public static final String GENERIC = "GENERIC";
    public static final String ANY = "GENERIC";
    public static final String NOUNPHRASE = "NP";
    public static final String NOUN = "NOUN";
	}

	/**
	 * Type of the annotation, e.g. PER, ORG, LOC
	 */
	protected String type;
  
  /**
   * Referenced ID, e.g. Wikipedia Page name
   */
  protected String refId;
  
	/**
	 * Creates a Typed MentionAnnotation using manual Span offsets.
	 */
	public MentionAnnotation(Source source, int begin, int end, String type, List<Token> tokens) {
		super(source, WordHelpers.wordsToText(tokens), begin, end);
		this.type = type;
	}
  
  /**
	 * Creates a GENERIC MentionAnnotation using manual Text and Span offsets.
	 */
  public MentionAnnotation(Source source, String text, int begin, int end) {
    super(source, text, begin, end);
    this.type = Type.GENERIC;
  }

  /**
   * Creates a GENERIC MentionAnnotation using Span offsets from given Tokens.
   */
  public MentionAnnotation(Source source, List<Token> tokens) {
    super(source, WordHelpers.tokensToText(tokens, tokens.get(0).getBegin()), 
            tokens.get(0).getBegin(), tokens.get(tokens.size()-1).getEnd());
    this.type = Type.GENERIC;
  }
  
  /**
   * Creates a GENERIC MentionAnnotation using Span offsets from given Tokens.
   */
  public MentionAnnotation(Source source, List<Token> tokens, double confidence) {
    super(source, WordHelpers.wordsToText(tokens), 
            tokens.get(0).getBegin(), tokens.get(tokens.size()-1).getEnd());
    this.setConfidence(confidence);
    this.type = Type.GENERIC;
  }
  
  /**
   * Creates a GENERIC MentionAnnotation using Span offsets from given Tokens.
   */
  public MentionAnnotation(Document doc, Source source, List<Token> tokens, String type, double confidence) {
    this(source, tokens, confidence);
    this.setDocumentRef(doc);
    this.type = (type == null || type.isEmpty()) ? Type.GENERIC : type;
  }
  
  /**
   * Default constructor.
   * @deprecated only used for JSON deserialization.
   */
  @Deprecated
  protected MentionAnnotation() {
    super();
  }
  
	/**
	 * @return the type of the Annotation e.g. PER, ORG, LOC
	 */
	public String getType() {
		return type;
	}
  
	/**
	 * Set the type of the Annotation
   * @param type
	 */
	public void setType(String type) {
		this.type = type;
	}
  
  /**
   * @return referenced ID, e.g. Wikipedia Page name
   */
  public String getRefId() {
    return refId;
  }

  public void setRefId(String id) {
    this.refId = id;
  }
  
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + begin;
		result = prime * result + end;
		result = prime * result + ((type == null) ? 0 : type.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
    if(!super.equals(obj)) return false;
		final MentionAnnotation other = (MentionAnnotation) obj;
		if(!this.matches(other)) return false;
		if(type == null) {
			if(other.type != null) return false;
		} else if(!type.equals(other.type)) return false;
    if(refId == null) {
			if(other.refId != null) return false;
		} else if(!refId.equals(other.refId)) return false;
    return true;
	}
  
	@Override
	public String toString() {
		return "Annotation [begin=" + begin + ", end=" + end + ", type=" + type + "]";
	}
  
  public static MentionAnnotation createFromPositions(Source source, Document doc, int begin, int end, String id, String type) {
    List<Token> tokens = new ArrayList<>();
    boolean inside = false;
    //boolean 
    for(Token t : doc.getTokens()) {
      if(!inside && t.getBegin() <= begin && t.getEnd() > begin) {
        //log.info("Found Begin Token for ann:" + id + " – " + t.getText());
        inside = true;
        t.putTag(Annotation.Source.GOLD, BIO2Tag.B());
        tokens.add(t);
      } else if(inside && t.getBegin() < end) {
        //log.info("Found Inside Token for ann:" + id + " – " + t.getText());
        t.putTag(Annotation.Source.GOLD, BIO2Tag.I());
        tokens.add(t);
      } else if(inside && t.getBegin() >= end) {
        //log.info("Found Outside Token for ann:" + id + " – " + t.getText());
        inside = false;
        MentionAnnotation ann = new MentionAnnotation(source, begin, end, type, tokens);
        ann.setRefId(id);
        ann.setDocumentRef(doc);
        return ann;
      }
    }
    if(inside) {
      MentionAnnotation ann = new MentionAnnotation(source, begin, end, type, tokens);
      ann.setRefId(id);
      ann.setDocumentRef(doc);
      return ann;
    } else {
      log.warn("Tokens not found for boundaries (" + begin + "," + end + "): " + doc.getText());
      return null;
    }
  }
  
  public static void annotateFromTags(Iterable<Document> docs, Annotation.Source source, Class<? extends Tag> tagset) {
    for(Document d : docs) annotateFromTags(d, source, tagset);
  }
  
  public static void annotateFromTags(Iterable<Document> docs, Annotation.Source source, Class<? extends Tag> tagset, String type) {
    for(Document d : docs) annotateFromTags(d, source, tagset, type);
  }
  
  public static void annotateFromTags(AbstractIterator it, Annotation.Source source, Class<? extends Tag> tagset) {
    annotateFromTags(it.getDocuments(), source, tagset);
  }
  
  public static void annotateFromTags(AbstractIterator it, Annotation.Source source, Class<? extends Tag> tagset, String type) {
    annotateFromTags(it.getDocuments(), source, tagset, type);
  }
  
  public static void annotateFromTags(Document doc, Annotation.Source source, Class<? extends Tag> tagset) {
    annotateFromTags(doc, source, tagset, MentionAnnotation.Type.GENERIC);
  }
  
  public static void annotateFromTags(Document doc, Annotation.Source source, Class<? extends Tag> tagset, String type) {
    List<? extends Annotation> anns = null;
    if(tagset.equals(BIO2Tag.class)) {
      anns = createFromBIO2Tags(doc, source, type);
    } else if(tagset.equals(BIOESTag.class)) {
      BIOESTag.convertToBIO2(doc, source);
      anns = createFromBIO2Tags(doc, source, type);
    } else {
      throw new IllegalArgumentException("Tagset " + tagset.getCanonicalName() + " not implemented");
    }
    if(anns != null) doc.addAnnotations(anns);
  }
  
  /**
   * requires: BIO2Tag.class on Token.class
   * attaches: MentionAnnotation.class to Document.class
   * @param doc
   * @param source
   * @return 
   */
  private static List<MentionAnnotation> createFromBIO2Tags(Document doc, Annotation.Source source, String defaultType) {
    List<MentionAnnotation> annotations = new ArrayList<>();
    List<Token> tokens = new ArrayList<>();
    
    double confidence = 0.;
    String type = defaultType;
    for(Sentence s : doc.getSentences()) {
      for(Token t : s.getTokens()) {
        BIO2Tag tag = t.getTag(source, BIO2Tag.class);
        if(tokens.isEmpty()) {
          if(tag.isB()) {
            tokens.add(t);
            confidence = tag.getConfidence();
            if(tag.getType() != null && !tag.getType().isEmpty()) type = tag.getType();
          } else if(tag.isI()) { // I after O, treat as B
            tokens.add(t);
            confidence = tag.getConfidence();
            if(tag.getType() != null && !tag.getType().isEmpty()) type = tag.getType();
          } else { // O after O
          }
        } else {
          if(tag.isB()) {
            annotations.add(new MentionAnnotation(s.getDocumentRef(), source, tokens, type, confidence / tokens.size()));
            tokens.clear();
            tokens.add(t);
            confidence = tag.getConfidence();
            if(tag.getType() != null && !tag.getType().isEmpty()) type = tag.getType();
          } else if(tag.isI()) {
            tokens.add(t);
            confidence += tag.getConfidence();
          } else {
            annotations.add(new MentionAnnotation(s.getDocumentRef(), source, tokens, type, confidence / tokens.size()));
            tokens.clear();
            confidence = 0.;
            type = defaultType;
          }
        }
      }
      if(!tokens.isEmpty()) { // last annotation
        annotations.add(new MentionAnnotation(s.getDocumentRef(), source, tokens, type, confidence / tokens.size()));
        tokens.clear();
      }
    }
    
    return annotations;
    
  }
  
  public static void createTagsFromAnnotations(Document doc, Annotation.Source source, Class<? extends Tag> tagset) {
    if(tagset.equals(BIOESTag.class)) {
      createBIOESTagsFromAnnotations(doc, source);
    } else if(tagset.equals(BIO2Tag.class)) {
      createBIOESTagsFromAnnotations(doc, source);
      BIOESTag.convertToBIO2(doc, source);
    } else {
      throw new IllegalArgumentException("Tagset " + tagset.getCanonicalName() + " not implemented");
    }
  }
  
  private static void createBIOESTagsFromAnnotations(Document doc, Annotation.Source source) {
    doc.streamAnnotations(source, MentionAnnotation.class).forEach(ann -> {
      List<Token> tokens = doc.streamTokensInRange(ann.getBegin(), ann.getEnd(), false).collect(Collectors.toList());
      if(tokens.isEmpty()) {
        log.warn("no Tokens in Annotation");
      } else if(tokens.size() == 1) {
        tokens.get(0).putTag(source, BIOESTag.S());
      } else {
        int i = 0;
        tokens.get(i).putTag(source, BIOESTag.B());
        while(++i < tokens.size() - 1) tokens.get(i).putTag(source, BIOESTag.I());
        tokens.get(i).putTag(source, BIOESTag.E());
      }
    });
  }
    
}
