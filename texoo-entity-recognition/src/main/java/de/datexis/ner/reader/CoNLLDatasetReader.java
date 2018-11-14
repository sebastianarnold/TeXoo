package de.datexis.ner.reader;

import de.datexis.preprocess.DocumentFactory;
import static de.datexis.common.WordHelpers.skipSpaceAfter;
import static de.datexis.common.WordHelpers.skipSpaceBefore;
import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import de.datexis.ner.MentionAnnotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Token;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.model.tag.BIO2Tag.Label;
import de.datexis.reader.DatasetReader;
import java.io.InputStream;

/**
 * Reads a Dataset from CoNLL formatted file.
 * @author sarnold
 */
public class CoNLLDatasetReader implements DatasetReader {

	private static final Logger log = LoggerFactory.getLogger(CoNLLDatasetReader.class);
  
  public static enum Charset { UTF_8, ISO_8859_1 };
  
  private static final String LINE_START = "-DOCSTART-";
  
  protected boolean useFirstSentenceAsTitle = false;
  
  protected Annotation.Source annotationSource = Annotation.Source.GOLD;
  
  protected int tagIndex = -1;
  
  protected String type = null;
  
  protected String name;
  
  /**
   * Use a specific name for the Dataset.
   */
  public CoNLLDatasetReader withName(String name) {
    this.name = name;
    return this;
  }
  
  /**
   * Use a specific column as NER tag.
   * @param index Use this index, starting from 0. Default: last column.
   */
  public CoNLLDatasetReader withTagIndex(int tagIndex) {
    this.tagIndex = tagIndex;
    return this;
  }
  
  /**
   * Use a copy of every first sentence as Document title.
   */
  public CoNLLDatasetReader withFirstSentenceAsTitle(boolean useFirstSentence) {
    this.useFirstSentenceAsTitle = useFirstSentence;
    return this;
  }
  
  /**
   * @param annotationSource Assign this source to all Annotations read from the file.
   */
  public CoNLLDatasetReader withAnnotationSource(Annotation.Source annotationSource) {
    this.annotationSource = annotationSource;
    return this;
  }
  
  /**
   * @param type Use this type instead of the given ones in the dataset,
   */
  public CoNLLDatasetReader withGenericType(String type) {
    this.type = type;
    return this;
  }
  
  /**
   * Read a Dataset from CoNLL file
   */
  @Override
  public Dataset read(Resource path) throws IOException {
    return read(path, Charset.UTF_8);
  }
  
  /**
   * Read a Dataset from CoNLL file
   */
	public Dataset read(Resource path, Charset charset) throws IOException {
    log.info("Reading Dataset from `{}`...", path.toString());
    Dataset data;
    try(InputStream in = path.getInputStream()) {
      CharsetDecoder cs;
      if(charset.equals(Charset.UTF_8)) cs = StandardCharsets.UTF_8.newDecoder();
      else cs = StandardCharsets.ISO_8859_1.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, cs));
      data = readLines(new LineIterator(br));
    }
    if(this.name != null) data.setName(name);
    else data.setName(path.getFileName().replaceFirst("\\..+$", ""));
    return data;
	}
  
  /**
   * Read a Dataset from CoNLL file (static version with default reader)
   */
  public static Dataset readDataset(Resource path, String name, Charset charset) throws IOException {
    final CoNLLDatasetReader reader = new CoNLLDatasetReader();
    Dataset data = reader.read(path, charset);
    data.setName(name);
    return data;
  }
  
  /**
	 * Create a Document from the given data
	 * @return - Document created from data
	 */
	protected Dataset readLines(Iterator<String> lines) {
    
		Dataset result = new Dataset();
		List<Token> tokens = new ArrayList<>();
		Token token = null;
		String type = null;

		int cursor = 0;
    String last = "";
		while(lines.hasNext()) {
			String line = lines.next().trim();
			if(line.startsWith(LINE_START)) {
        // end document
        if(!tokens.isEmpty()) {
          Document document = DocumentFactory.fromTokens(tokens);
          MentionAnnotation.annotateFromTags(document, annotationSource, BIO2Tag.class);
          result.addDocument(document);
        }
        // start new document
        cursor = 0;
				tokens = new ArrayList<>();
        type = null;
        last = "";
      } else if(line.length() == 0) {
        // end sentence
        if(!tokens.isEmpty()) {
          tokens.add(new Token("\n"));
          cursor++;
          last = token.getText();
        }
        type = null;
      } else if(line.length() > 0) {
        // read token
        token = createTokenFromLine(line, cursor, type);
				if(token != null) {
					tokens.add(token);
          cursor = token.getEnd();
          if(!skipSpaceAfter.contains(last) && !skipSpaceBefore.contains(token.getText())) cursor++;
					type = token.getTag(annotationSource, BIO2Tag.class).getType();
          last = token.getText();
				}
      }
		}
		// end document
    if(!tokens.isEmpty()) {
      Document document = DocumentFactory.fromTokens(tokens);
      MentionAnnotation.annotateFromTags(document, annotationSource, BIO2Tag.class);
      result.addDocument(document);
    }
    
    for(Document doc : result.getDocuments()) {
      if(useFirstSentenceAsTitle) {
        if(doc.countSentences() > 0) {
          doc.setTitle(doc.getSentence(0).getText());
        } else {
          doc.setTitle("");
        }
      }
      doc.setTagAvailable(annotationSource, BIO2Tag.class, true);
    }
    
    log.info(String.format("Finished reading dataset (%,d docs, %,d sentences, %,d tokens, %,d mentions)", 
            result.countDocuments(), result.countSentences(), result.countTokens(), result.countAnnotations()));
    
		return result;
	}
   
	/**
	 * Creates Token from the given line of CoNLL2003 data
	 * @param line - CoNLL2003 data to create Token
	 * @param index - character index in the whole document
	 * @return Token created from line
	 */
	protected Token createTokenFromLine(String line, int cursor, String prevType) {
		try {
      String[] csv = line.split("\\s+");
      int pos = tagIndex >= 0 ? tagIndex : csv.length - 1;
			String text = csv[0];
			BIO2Tag tag = createTag(csv[pos], prevType);
			int start = cursor;
			int end = cursor + text.length();
			Token token = new Token(text, start, end);
			token.putTag(annotationSource, tag);
			return token;
		} catch (Exception e) {
			log.warn("could not read line: " + line);
			return null;
		}
	}

  /**
   * Returns the NER Label based on current and previous tags
   * @param tag
   * @param prevType
   * @return 
   */
  protected BIO2Tag createTag(String label, String prevType) {
		String[] parts = label.split("\\-");
		String tag = parts[0];
		String type;
    if(this.type != null) {
      type = this.type;
    } else {
      type = parts.length > 1 ? parts[1] : MentionAnnotation.Type.GENERIC;
    }
    switch(tag) {
      case "O":
        return new BIO2Tag(Label.O, null);
      case "B":
        return new BIO2Tag(Label.B, type);
      case "I":
        if(type.equals(prevType)) return new BIO2Tag(Label.I, type);
        else return new BIO2Tag(Label.B, type);
      default:
        log.warn("reading unknown tag " + label);
        return new BIO2Tag(Label.O, null);
		}
	}
  
}
