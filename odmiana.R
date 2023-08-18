# library(xml2)
# pliki <- c('a-publi.xml', 'b-prasa.xml', 'c-popul.xml', 'd-proza.xml', 'e-dramat.xml')
# pliki <- paste0('pl196x/', pliki)
# df <- data.frame(odmienione=c(), slownikowe=c(), morfologia=c(), stringsAsFactors = F)
# for(plik in pliki){
#   dane <- xml2::read_xml(plik)
#   dane <- xml2::xml_child(dane,'text')
#   dane <- xml2::xml_child(dane,'group')
#   dane <- xml2::xml_contents(dane)
#   dane <- xml2::xml_child(dane,'body')
#   dane <- xml2::xml_contents(dane)
#   dane <- xml2::xml_contents(dane)
#   dane <- xml2::xml_contents(dane)
#   slowa <- xml_text(dane)
#   atr <- xml_attrs(dane)
#   atr1 <- sapply(atr, function(x) x[2])
#   atr2 <- sapply(atr, function(x) x[3])
#   atr2 <- strsplit(atr2, ' ')
#   atr2 <- sapply(atr2, function(x) x[1])
#   df_pom <- data.frame(odmienione=slowa, slownikowe=atr1, morfologia=atr2, stringsAsFactors = F)
#   df <- rbind(df, df_pom)
# }
# df <- na.omit(df)
# hist(nchar(df[,1]))
# morf <- read_xml('pl196x/fslib.xml')
# morf <- xml2::xml_contents(morf)
# atr <- simplify2array(xml_attrs(morf))
# atr <- t(atr)
# morf <- simplify2array(strsplit(atr[,2], '               '))
# morf <- t(morf)
# rownames(morf) <- atr[,1]
#
#
# morf2 <- sapply(df[,3], function(x) morf[x,])
# morf2 <- t(morf2)
# df <- cbind(df[1:2], morf2)
# colnames(df)[3:ncol(df)] <- c('czescMowy', 'liczba', 'przypadek', 'rodzaj',
#                               'stopien', 'osoba', 'czas', 'tryb', 'aspekt',
#                               'strona', 'akcent', 'poprzyimk', 'oznCzas', 'kategoria')
#
# dousuniecia <- which(df$czescMowy %in% c('przyimek', 'partykula', 'spojnik', 'kategoriaNieznana', 'wykrzyknik'))
# df <- df[-dousuniecia,]
#
# write.csv2(df, 'polski.csv', quote=F, row.names = F, fileEncoding = 'UTF-8')
#
# remove(atr, atr2, atr1, slowa, dane, df_pom, plik, pliki, morf, morf2, dousuniecia)

library(keras)

maks_liter <- 16
n <- 75000

df <- read.csv2('polski.csv', colClasses = c(rep('character',2), rep('factor',14)), encoding = 'UTF-8')
#hist(nchar(df[,1]))
df <- df[which(nchar(df[,1])<=maks_liter),]
df <- df[which(nchar(df[,2])<=maks_liter),]
id <- sample.int(nrow(df), n)
df <- df[id,]

dane <- df[,2]
wz <- df[,1]
morf <- df[3:16]
for(i in 1:ncol(morf)){
  morf[,i] <- as.numeric(morf[,i])
}
morf <- as.matrix(morf)

tokenizer <- text_tokenizer(char_level = T)
tokenizer %>% fit_text_tokenizer(dane)
dane <- tokenizer %>% texts_to_sequences(dane)
wz <- tokenizer %>% texts_to_sequences(wz)
dane <- pad_sequences(dane)
wz <- pad_sequences(wz)
zn <- unlist(tokenizer$index_word, use.names = F)
l_zn <- length(zn)+1
dim(dane) <- c(dim(dane),1)
dim(wz) <- c(dim(wz),1)

id_wal <- sample.int(dim(dane)[1], 100)
dane_wal <- dane[id_wal,,,drop=F]
wz_wal <- wz[id_wal,,,drop=F]
morf_wal <- morf[id_wal,]
dane <- dane[-id_wal,,,drop=F]
wz <- wz[-id_wal,,,drop=F]
morf <- morf[-id_wal,]

# id_test <- sample.int(dim(dane)[1], 0.15*dim(dane)[1])
# dane_test <- dane[id_test,,]
# wz_test <- wz[id_test,]
# morf_test <- morf[id_test,]
# dane <- dane[-id_test,,,drop=F]
# wz <- wz[-id_test,]
# morf <- morf[-id_test,]

neurony1 <- 512
neurony2 <- 512

morf_we <- layer_input(ncol(morf))
morf_mpl1 <- layer_dense(morf_we, neurony1, 'tanh')
morf_mpl2 <- layer_dense(morf_we, neurony1, 'tanh')
dane_we <- layer_input(c(dim(dane)[2],1))
enkoder <- layer_lstm(units = neurony1)
enkoder_wy <- enkoder(dane_we, initial_state = list(morf_mpl1, morf_mpl2))
enkoder_rep <- layer_repeat_vector(enkoder_wy, n=ncol(wz))
dekoder <- layer_lstm(enkoder_rep, units = neurony2, return_sequences = T)
wy <- layer_dense(dekoder, l_zn, 'softmax')

model <- keras_model(list(dane_we, morf_we), wy)
model %>% compile(optimizer_adam(lr=0.003), 'sparse_categorical_crossentropy', list('acc'))
model %>% fit(list(dane, morf), wz, epochs=20, validation_split = 0.15, view_metrics=F, batch_size = 512,
              callbacks = list(callback_early_stopping(patience=7, restore_best_weights = T)))

model %>% evaluate(list(dane_wal, morf_wal), wz_wal)
pred <- model %>% predict(list(dane_wal, morf_wal))
for(i in 1:length(id_wal)){
  cat('Jest: ', paste0(zn[apply(pred[i,,], 1, which.max)-1], collapse = ''),'\n')
  cat('Mialo byc:', df[id_wal[i],1],'\n')
  cat('---------------------\n')
}
save_model_hdf5(model, 'odmiana1')
