library(dplyr)
library(tm)
library(caret)

# Read XML document
raw.file = "../../data/qualys/latest.qkdb.xml.zip"
doc <- xml2::read_xml(raw.file)

# Extract QID, SEVERITY_LEVEL and DIAGNOSIS
kdb_txt <- rvest::html_text(rvest::html_elements(doc, xpath="//VULN[DIAGNOSIS]/*[self::QID or self::SEVERITY_LEVEL or self::DIAGNOSIS]"))
kdb_txt <- matrix(kdb_txt, nrow = length(kdb_txt)/3, ncol = 3, byrow = TRUE)
kdb_txt <- as.data.frame.matrix(kdb_txt)
names(kdb_txt) <- c("qid", "severity", "diagnosis")

# Tidy data frame
kdb_txt$qid <- as.integer(kdb_txt$qid)
kdb_txt$severity <- as.integer(kdb_txt$severity)
kdb_txt$diagnosis <- textclean::replace_html(kdb_txt$diagnosis)
kdb_txt$critical <- ifelse(test = kdb_txt$severity < 5, yes = "NO", no = "YES")
kdb_txt$criticalb <- kdb_txt$severity == 5

# Text analysis
## Stopwords
freq_word <- sort(table(unlist(strsplit(kdb_txt$diagnosis, " "))), decreasing = TRUE)
kdb_words <- names(freq_word)[(which(!(names(freq_word) %in% stopwords::stopwords())))]
## Characters
freq_char <- sort(table(unlist(strsplit(kdb_txt$diagnosis, ""))), decreasing = TRUE)

kdb_txt$descr <- textclean::replace_symbol(kdb_txt$diagnosis)
freq_char2 <- sort(table(unlist(strsplit(kdb_txt$descr, ""))), decreasing = TRUE)
freq_char2

# Prepare data for training
kdb_critical <- kdb_txt %>% filter(critical == "YES")
kdb_other <- kdb_txt %>% filter(critical == "NO")

kdb_ml <- bind_rows(kdb_critical %>% sample_n(2000),
                    kdb_other %>% sample_n(2000)) %>%
          sample_n(4000) %>%
          select(descr, critical)
table(kdb_ml$critical)

#*******************************************************************
#                         Classification
#*******************************************************************
# install.packages("tm")
#-------------------------------------------------------------------
#                  4.2.: Preparing data for Classification
#-------------------------------------------------------------------
#Load up the corpus
# course_raw = scan("data/Course-Descriptions.txt", what="", sep="\n")
course_raw <- kdb_ml$descr
course_corpus <- VCorpus(VectorSource(course_raw))
inspect(course_corpus[[1]])
#Convert to lower case
course_corpus2 <- tm_map(course_corpus, content_transformer(tolower))
#Remove punctuations
course_corpus3 <- tm_map(course_corpus2, removePunctuation)
#Remove stopwords
course_corpus4 <- tm_map(course_corpus3, removeWords, stopwords())
inspect(course_corpus4[[1]])
#Generate TF-IDF matrix
course_dtm <- DocumentTermMatrix(course_corpus4)
course_dtm
findFreqTerms(course_dtm,5)
#Remove terms not in 90% of the documents. Only have those that are there
#in atleast 2 documents
dense_course_dtm <- removeSparseTerms(course_dtm, .85)
#Inspect to TF-IDF
inspect(dense_course_dtm)
#Convert continuous values to classes = { Yes, No }
conv_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}
class_dtm <- apply(dense_course_dtm, MARGIN = 2, conv_counts)
class_dtm
#-------------------------------------------------------------------
#                  4.3.: Building the model
#-------------------------------------------------------------------
#Load the classifications for the descriptions
# course_classes = scan("data/Course-Classification.txt", what="", sep="\n")
course_classes <- kdb_ml$critical
#install.packages("caret")
#Random split of training and testing sets
train_set <- createDataPartition(y=course_classes, p=.7,list=FALSE)
#spliting the dtm
train_dtm <- class_dtm[train_set,]
test_dtm <-class_dtm[-train_set,]
#split the course_classes
train_classes <- course_classes[train_set]
test_classes <- course_classes[-train_set]
#train the model using naive bayes
course_model <- train( data.frame(train_dtm), train_classes, method="nb")
course_model

#-------------------------------------------------------------------
#                  4.3.: Predictions for Text
#-------------------------------------------------------------------
#Predict for the test data
course_predictions <- predict(course_model,test_dtm)
#Analyze prediction accuracy
confusionMatrix(table(course_predictions , test_classes))
#-------------------------------------------------------------------

